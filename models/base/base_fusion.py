import torch
import torch.nn.functional as F
from torch import optim
import pandas as pd
import os
from sklearn.metrics import (
    average_precision_score, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score
)
import numpy as np
import lightning as L
from utils.feature_saver import FeatureSaver
from utils.fairness_metrics import compute_fairness_metrics
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

# BaseFuseTrainer: Base class for multimodal clinical prediction training and evaluation
class BaseFuseTrainer(L.LightningModule):
    """
    BaseFuseTrainer

    This is the base class for training and evaluating multimodal clinical prediction models.
    It extends LightningModule from PyTorch Lightning and is intended to be subclassed
    by specific fusion models.

    Main Features:
    ---------------
    - Defines the foundational infrastructure for handling the training and evaluation workflow 
      in multimodal (EHR + imaging, etc.) clinical prediction tasks.
    - Manages stateful objects and caches for training, validation, and testing, including
      predictions, labels, and modality-specific predictions.
    - Supports batch data device placement and forward propagation logic.
    - Handles label weights initialization, feature saving for each split (train/val/test),
      and storing/loading checkpoint paths.
    - Implements utility functions for performance evaluation, monitoring PR AUC, and managing
      fairness-related caches.
    - Includes generic methods that are meant to be overridden by child classes that define
      actual model logic and data-specific forward propagation.

    Usage:
    -------
    Subclass this base to define concrete multimodal fusion architectures and supply dataset/task-
    specific forward step implementations.

    Example:
    --------
    class MyMultimodalNetwork(BaseFuseTrainer):
        def __init__(self, ...):
            super().__init__()
            # Define custom layers, modality fusion strategy, etc.

        def forward(self, data_dict):
            # Implement fusion and output prediction logic
            pass
    -----------
    """
    def __init__(self):
        super().__init__()
        # Store training state and caches for each split
        self.train_info = {}
        self.checkpoint_path = ""
        self.max_prauc = -1
        # Cache for predictions, labels, modality-specific predictions, group and meta info
        self.val_info = {'predictions': [], 'labels': [], 'pred_ehr': [], 'pred_cxr': [], 'groups': [], 'meta_attrs': []}
        self.test_info = {'predictions': [], 'labels': [], 'pred_ehr': [], 'pred_cxr': [], 'groups': [], 'meta_attrs': []}
        self.feature_saver = None
        self.total_time = 0
        self.epoch_start_time = None
        self._init_label_weights()  # Initialize label weights

    # Initialize and log label weights (if provided)
    def _init_label_weights(self):
        if hasattr(self, 'hparams') and hasattr(self.hparams, 'label_weights') and self.hparams.label_weights is not None:
            if isinstance(self.hparams.label_weights, (list, tuple)):
                self.hparams.label_weights = torch.tensor(self.hparams.label_weights, dtype=torch.float32)
            if isinstance(self.hparams.label_weights, torch.Tensor):
                self.hparams.label_weights_for_logging = self.hparams.label_weights.detach().cpu().numpy().tolist()
            else:
                self.hparams.label_weights_for_logging = self.hparams.label_weights
            print(f"Label weights initialized: {self.hparams.label_weights}")
        else:
            print("Label weights not provided, will use default weights")

    # Move batch data to appropriate device
    def __get_batch_data(self, batch):
        for x in ['ehr_ts', 'cxr_imgs', 'labels']:
            batch[x] = batch[x].to(self.device)
        return batch

    def forward(self, data_dict):
        # Must be implemented by child classes
        raise NotImplementedError('The `forward` method must be implemented in child classes.')

    # Common step for forward propagation and optional feature saving
    def _shared_step(self, batch):
        batch = self.__get_batch_data(batch)
        out = self(batch)
        if getattr(self.hparams, 'save', False) and 'feat_ehr_distinct' in out:
            # Add features for saving if enabled
            split = 'train' if self.training else 'val' if self.training_type == 'val' else 'test'
            self.feature_saver.add_features(
                split,
                out['feat_ehr_distinct'].detach().cpu().numpy(),
                out['feat_cxr_distinct'].detach().cpu().numpy(),
                batch['labels'].detach().cpu().numpy()
            )
        return out

    # Training step: forward + loss computation + logging
    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.log_dict({'loss/train': out['loss'].detach()}, on_epoch=True, on_step=True, batch_size=batch['labels'].shape[0], sync_dist=True)
        return out['loss']

    # Shared validation/test logic, fills cache for evaluation
    def _val_test_shared_step(self, batch, cache):
        out = self._shared_step(batch)
        cache['predictions'].append(out['predictions'].detach())
        cache['labels'].append(batch['labels'].detach())
        if 'meta_attrs' in batch:
            if 'meta_attrs' not in cache:
                cache['meta_attrs'] = []
            meta_attrs_list = batch['meta_attrs'].to_dict('records')
            cache['meta_attrs'].extend(meta_attrs_list)
        if 'groups' in batch:
            cache['groups'].extend(batch['groups'])
        return out

    # End-of-epoch hook: evaluates performance and clears cache
    def _val_test_epoch_end(self, cache, clear_cache=True):
        meta_attrs_df = None
        if cache.get('meta_attrs'):
            meta_attrs_df = pd.DataFrame(cache['meta_attrs'])
        scores = self.evaluate_performance(
            torch.cat(cache['predictions']), 
            torch.cat(cache['labels']),
            meta_attrs=meta_attrs_df
        )
        if clear_cache:
            for x in cache:
                cache[x] = [] if isinstance(cache[x], list) else []
        return scores

    # Runs shared validation logic and logs filtered overall scores
    def validation_step(self, batch, batch_idx):
        out = self._val_test_shared_step(batch, self.val_info)
        self.log_dict({'loss/validation': out['loss'].detach()}, on_epoch=True, on_step=True, batch_size=batch['labels'].shape[0], sync_dist=True)
        return out['loss'].detach()

    # Only fills test_info cache
    def test_step(self, batch, batch_idx):
        self._val_test_shared_step(batch, self.test_info)

    # End validation: save features and log only overall/aggregate metrics
    def on_validation_epoch_end(self):
        if getattr(self.hparams, 'save', False):
            self.feature_saver.save_features('val', self.current_epoch, getattr(self.hparams, 'hidden_size', 256))
        scores = self._val_test_epoch_end(self.val_info, clear_cache=True)
        # Log only overall metrics to avoid too many CSV columns
        filtered_scores = {
            k: v for k, v in scores.items() 
            if not isinstance(v, (list, str)) and (
                k.startswith('overall/') or 
                k.startswith('loss/') or 
                k.startswith('train_loss/') or 
                k.startswith('disentangle_train/') or
                k.startswith('system/')
            )
        }
        self.log_dict(filtered_scores, on_epoch=True, on_step=False, sync_dist=True)

    # End test: save features/predictions and store all test results
    def on_test_epoch_end(self):
        if getattr(self.hparams, 'save', False):
            self.feature_saver.save_features('test', self.current_epoch, getattr(self.hparams, 'hidden_size', 256))
        if getattr(self.hparams, 'save_predictions', False):
            self._save_test_predictions_and_labels()
        scores = self._val_test_epoch_end(self.test_info, clear_cache=True)
        self.test_results = scores
    
    # Save test predictions, labels, meta info to disk
    def _save_test_predictions_and_labels(self):
        if not self.test_info['predictions'] or not self.test_info['labels']:
            print("No test predictions or labels to save.")
            return
        predictions = torch.cat(self.test_info['predictions']).cpu().numpy()
        labels = torch.cat(self.test_info['labels']).cpu().numpy()
        save_dir = None
        # Try to get experiment directory from logger
        if hasattr(self, 'logger') and self.logger is not None:
            if isinstance(self.logger, list):
                for logger in self.logger:
                    if hasattr(logger, 'log_dir'):
                        save_dir = logger.log_dir
                        break
            elif hasattr(self.logger, 'log_dir'):
                save_dir = self.logger.log_dir
        if save_dir is None:
            save_dir = getattr(self.hparams, 'predictions_save_dir', './test_predictions')
            print(f"Warning: Could not get experiment log directory, using fallback: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        model_name = getattr(self.hparams, 'model_name', 'unknown_model')
        task = getattr(self.hparams, 'task', 'unknown_task')
        fold = getattr(self.hparams, 'fold', 0)
        seed = getattr(self.hparams, 'seed', 0)
        npz_filename = f"test_predictions_fold{fold}_seed{seed}.npz"
        csv_filename = f"test_predictions_fold{fold}_seed{seed}.csv"
        npz_path = os.path.join(save_dir, npz_filename)
        csv_path = os.path.join(save_dir, csv_filename)
        save_data = {
            'predictions': predictions,
            'labels': labels,
            'model_name': model_name,
            'task': task,
            'fold': fold,
            'seed': seed,
            'predictions_shape': predictions.shape,
            'labels_shape': labels.shape
        }
        # Save extra meta information if available
        if self.test_info.get('meta_attrs'):
            meta_attrs_df = pd.DataFrame(self.test_info['meta_attrs'])
            for col in meta_attrs_df.columns:
                save_data[f'meta_{col}'] = meta_attrs_df[col].values
        if self.test_info.get('pred_ehr'):
            pred_ehr = torch.cat(self.test_info['pred_ehr']).cpu().numpy()
            save_data['pred_ehr'] = pred_ehr
        if self.test_info.get('pred_cxr'):
            pred_cxr = torch.cat(self.test_info['pred_cxr']).cpu().numpy()
            save_data['pred_cxr'] = pred_cxr
        np.savez_compressed(npz_path, **save_data)
        # Prepare CSV columns for predictions, labels, modalities, and meta data
        csv_data = {}
        if predictions.ndim == 1:
            csv_data['predictions'] = predictions
        elif predictions.ndim == 2 and predictions.shape[1] == 1:
            csv_data['predictions'] = predictions.flatten()
        else:
            for i in range(predictions.shape[1]):
                csv_data[f'pred_class_{i}'] = predictions[:, i]
        if labels.ndim == 1:
            csv_data['labels'] = labels
        elif labels.ndim == 2 and labels.shape[1] == 1:
            csv_data['labels'] = labels.flatten()
        else:
            for i in range(labels.shape[1]):
                csv_data[f'label_class_{i}'] = labels[:, i]
        if self.test_info.get('meta_attrs'):
            for col in meta_attrs_df.columns:
                csv_data[f'meta_{col}'] = meta_attrs_df[col].values
        if self.test_info.get('pred_ehr'):
            pred_ehr = torch.cat(self.test_info['pred_ehr']).cpu().numpy()
            if pred_ehr.ndim == 1:
                csv_data['pred_ehr'] = pred_ehr
            else:
                for i in range(pred_ehr.shape[1]):
                    csv_data[f'pred_ehr_class_{i}'] = pred_ehr[:, i]
        if self.test_info.get('pred_cxr'):
            pred_cxr = torch.cat(self.test_info['pred_cxr']).cpu().numpy()
            if pred_cxr.ndim == 1:
                csv_data['pred_cxr'] = pred_cxr
            else:
                for i in range(pred_cxr.shape[1]):
                    csv_data[f'pred_cxr_class_{i}'] = pred_cxr[:, i]
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"Test predictions and labels saved to experiment directory:")
        print(f"   Experiment dir: {save_dir}")
        print(f"   NPZ format: {npz_filename}")
        print(f"   CSV format: {csv_filename}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Total samples: {len(predictions)}")
    
    # Evaluate model performance (binary/multilabel/multiclass auto-detect)
    def evaluate_performance(self, preds, labels, meta_attrs=None):
        y_true = labels.cpu().numpy()
        y_score = preds.cpu().numpy()
        if hasattr(self.hparams, 'task'):
            task_type = self.hparams.task
        else:
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                task_type = 'phenotype'
            elif y_score.shape[-1] > 2:
                task_type = 'los'
            else:
                task_type = 'mortality'
        print(f"\nEvaluating {task_type} task...")
        if task_type == 'phenotype':
            results = self._evaluate_multilabel(y_true, y_score)
        elif task_type == 'los':
            results = self._evaluate_multiclass(y_true, y_score)
        else:
            results = self._evaluate_binary(y_true, y_score)
        # Print info on fairness metric eligibility
        print("Checking fairness computation conditions...")
        print(f"   compute_fairness: {getattr(self.hparams, 'compute_fairness', False)}")
        print(f"   meta_attrs is not None: {meta_attrs is not None}")
        if meta_attrs is not None:
            print(f"   meta_attrs length: {len(meta_attrs)}")
            print(f"   meta_attrs columns: {list(meta_attrs.columns) if hasattr(meta_attrs, 'columns') else 'No columns'}")
        # Condition to compute fairness: flag enabled and meta_attrs present and nonempty
        if (getattr(self.hparams, 'compute_fairness', False) and 
            meta_attrs is not None and len(meta_attrs) > 0):
            print("Fairness computation conditions met, computing fairness metrics...")
            try:
                fairness_results = self._compute_fairness_metrics(
                    y_true, y_score, meta_attrs, task_type
                )
                results.update(fairness_results)
            except Exception as e:
                print(f"Warning: Failed to compute fairness metrics: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Fairness computation conditions not met")
        return results

    # Call unified fairness metrics computation for any task type
    def _compute_fairness_metrics(self, y_true, y_score, meta_attrs, task_type):
        sensitive_attributes = getattr(self.hparams, 'fairness_attributes', ['race', 'gender', 'age'])
        age_bins = getattr(self.hparams, 'fairness_age_bins', [0, 40, 60, 80])
        age_bins.append(float('inf'))
        compute_intersectional = getattr(self.hparams, 'fairness_intersectional', False)
        include_cxr_availability = getattr(self.hparams, 'fairness_include_cxr', True)
        fairness_task_type = task_type
        if task_type == 'mortality':
            fairness_task_type = 'binary'
        elif task_type == 'los':
            fairness_task_type = 'multiclass'
        elif task_type == 'phenotype':
            fairness_task_type = 'multilabel'
        return compute_fairness_metrics(
            y_true=y_true,
            y_score=y_score,
            meta_attrs=meta_attrs,
            task_type=fairness_task_type,
            sensitive_attributes=sensitive_attributes,
            age_bins=age_bins,
            compute_intersectional=compute_intersectional,
            include_cxr_availability=include_cxr_availability
        )

    # Evaluation for binary classification tasks
    def _evaluate_binary(self, y_true, y_score):
        y_true = y_true.flatten()
        y_score = y_score.flatten()
        y_binarized = (y_score > 0.5).astype(int)
        pos_label = 1
        results = {
            'overall/PRAUC': float(average_precision_score(y_true, y_score)),
            'overall/AUROC': float(roc_auc_score(y_true, y_score)),
            'overall/ACC': float(accuracy_score(y_true, y_binarized)),
            'overall/F1': float(f1_score(y_true, y_binarized)),
            'overall/Precision': float(precision_score(y_true, y_binarized)),
            'overall/Recall': float(recall_score(y_true, y_binarized, pos_label=pos_label)),
            'overall/Specificity': float(recall_score(y_true, y_binarized, pos_label=1 - pos_label)),
        }
        print(f"\nBinary Classification Results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        return results

    # Evaluation for multilabel (phenotype) classification
    def _evaluate_multilabel(self, y_true, y_score):
        binarized = (y_score > 0.5).astype(int)
        num_classes = y_true.shape[1]
        pos_label = 1
        if hasattr(self.hparams, 'class_names') and self.hparams.class_names is not None:
            class_names = self.hparams.class_names
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        results = {}
        for i in range(num_classes):
            name = class_names[i]
            y_t = y_true[:, i]
            y_s = y_score[:, i]
            y_b = binarized[:, i]
            try:
                results[f'PRAUC/{name}'] = float(average_precision_score(y_t, y_s))
                results[f'AUROC/{name}'] = float(roc_auc_score(y_t, y_s))
                results[f'ACC/{name}'] = float(accuracy_score(y_t, y_b))
                results[f'F1/{name}'] = float(f1_score(y_t, y_b))
                results[f'Precision/{name}'] = float(precision_score(y_t, y_b))
                results[f'Recall/{name}'] = float(recall_score(y_t, y_b, pos_label=pos_label))
                results[f'Specificity/{name}'] = float(recall_score(y_t, y_b, pos_label=1 - pos_label))
            except ValueError:
                print(f"[Warning] Skipping class {name} due to metric error.")
                continue
        # Compute mean for each metric across all classes
        for metric in ['PRAUC', 'AUROC', 'ACC', 'F1', 'Precision', 'Recall', 'Specificity']:
            values = [v for k, v in results.items() if k.startswith(metric + '/')]
            if values:
                results[f'overall/{metric}'] = float(np.mean(values))
        # Weighted and micro metrics
        try:
            supports = []
            prauc_values = []
            auroc_values = []
            for i in range(num_classes):
                y_t = y_true[:, i]
                y_s = y_score[:, i]
                support = np.sum(y_t)
                if support > 0 and support < len(y_t):
                    try:
                        prauc = average_precision_score(y_t, y_s)
                        auroc = roc_auc_score(y_t, y_s)
                        supports.append(support)
                        prauc_values.append(prauc)
                        auroc_values.append(auroc)
                    except ValueError:
                        continue
            if supports and prauc_values and auroc_values:
                total_support = sum(supports)
                if total_support > 0:
                    weighted_prauc = sum(p * s for p, s in zip(prauc_values, supports)) / total_support
                    weighted_auroc = sum(a * s for a, s in zip(auroc_values, supports)) / total_support
                    results['overall/PRAUC_weighted'] = float(weighted_prauc)
                    results['overall/AUROC_weighted'] = float(weighted_auroc)
        except Exception as e:
            print(f"[Warning] Could not calculate weighted PRAUC/AUROC: {e}")
        try:
            micro_prauc = average_precision_score(y_true.ravel(), y_score.ravel())
            micro_auroc = roc_auc_score(y_true.ravel(), y_score.ravel())
            results['overall/PRAUC_micro'] = float(micro_prauc)
            results['overall/AUROC_micro'] = float(micro_auroc)
        except Exception as e:
            print(f"[Warning] Could not calculate micro PRAUC/AUROC: {e}")
        print(f"\nMulti-label Classification Results:")
        for k, v in results.items():
            if k.startswith('overall/'):
                print(f"{k}: {v:.4f}")
        return results

    # Evaluation for multiclass classification (e.g. LoS)
    def _evaluate_multiclass(self, y_true, y_score):
        y_true = y_true.flatten().astype(int)
        if y_score.ndim > 1 and y_score.shape[1] > 1:
            y_pred = np.argmax(y_score, axis=1)
        else:
            y_pred = y_score.flatten().astype(int)
        results = {
            'overall/ACC': float(accuracy_score(y_true, y_pred)),
            'overall/F1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'overall/F1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
            'overall/Precision_macro': float(precision_score(y_true, y_pred, average='macro')),
            'overall/Precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
            'overall/Recall_macro': float(recall_score(y_true, y_pred, average='macro')),
            'overall/Recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
        }
        kappa_score = cohen_kappa_score(y_true, y_pred)
        results['overall/Kappa'] = float(kappa_score)
        print(f"Cohen's Kappa Score: {kappa_score:.4f}")
        los_bin_names = ["2-3d", "3-4d", "4-5d", "5-6d", "6-7d", "7-14d", "14+d"]
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        specificity_values = []
        support_values = []
        # Compute per-class metrics (F1, Precision, Recall, Specificity, Support)
        for cls in unique_classes:
            if cls < len(los_bin_names):
                cls_name = los_bin_names[cls]
            else:
                cls_name = f"Class_{cls}"
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            if np.sum(y_true_binary) > 0:
                try:
                    results[f'F1/{cls_name}'] = float(f1_score(y_true_binary, y_pred_binary))
                    results[f'Precision/{cls_name}'] = float(precision_score(y_true_binary, y_pred_binary))
                    results[f'Recall/{cls_name}'] = float(recall_score(y_true_binary, y_pred_binary))
                    results[f'Support/{cls_name}'] = float(np.sum(y_true_binary))
                    specificity = recall_score(y_true_binary, y_pred_binary, pos_label=0)
                    results[f'Specificity/{cls_name}'] = float(specificity)
                    specificity_values.append(specificity)
                    support_values.append(np.sum(y_true_binary))
                except ValueError:
                    print(f"[Warning] Skipping metrics for class {cls_name} due to computation error.")
                    continue
        if specificity_values:
            results['overall/Specificity_macro'] = float(np.mean(specificity_values))
            if support_values and len(specificity_values) == len(support_values):
                total_support = sum(support_values)
                weighted_specificity = sum(spec * supp for spec, supp in zip(specificity_values, support_values)) / total_support
                results['overall/Specificity_weighted'] = float(weighted_specificity)
        # Print confusion matrix and additional info
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nLoS Prediction Confusion Matrix:")
        print("True\\Pred", end="")
        for i in range(min(len(los_bin_names), cm.shape[1])):
            print(f"\t{los_bin_names[i][:6]}", end="")
        print()
        for i in range(cm.shape[0]):
            if i < len(los_bin_names):
                print(f"{los_bin_names[i][:8]}", end="")
            else:
                print(f"Class_{i}", end="")
            for j in range(cm.shape[1]):
                print(f"\t{cm[i,j]}", end="")
            print()
        print(f"\nLoS Class Distribution:")
        unique, counts = np.unique(y_true, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < len(los_bin_names):
                cls_name = los_bin_names[cls]
            else:
                cls_name = f"Class_{cls}"
            percentage = count / len(y_true) * 100
            print(f"  {cls_name}: {count} samples ({percentage:.2f}%)")
        print(f"\nMulti-class Classification Results:")
        print(f"\nOverall Metrics:")
        overall_metrics = [k for k in results.keys() if k.startswith('overall/')]
        for metric in sorted(overall_metrics):
            print(f"  {metric}: {results[metric]:.4f}")
        print(f"\nPer-Class Metrics:")
        per_class_metrics = [k for k in results.keys() if not k.startswith('overall/')]
        metric_types = ['F1', 'Precision', 'Recall', 'Specificity', 'Support']
        for metric_type in metric_types:
            type_metrics = [k for k in per_class_metrics if k.startswith(f'{metric_type}/')]
            if type_metrics:
                print(f"\n  {metric_type} by Class:")
                for metric in sorted(type_metrics):
                    print(f"    {metric}: {results[metric]:.4f}")
        return results

    # Evaluate group-wise fairness metrics (support for los/binary/multilabel)
    def evaluate_fairness(self, group_list, preds, labels):
        y_true = labels.cpu().numpy()
        y_score = preds.cpu().numpy()
        group_arr = np.array(group_list)
        scores = {}
        for group in np.unique(group_arr):
            mask = (group_arr == group)
            if np.sum(mask) == 0:
                continue
            try:
                if hasattr(self.hparams, 'task') and self.hparams.task == 'los':
                    y_true_group = y_true[mask].flatten().astype(int)
                    if y_score.ndim > 1 and y_score.shape[1] > 1:
                        y_pred_group = np.argmax(y_score[mask], axis=1)
                    else:
                        y_pred_group = y_score[mask].flatten().astype(int)
                    acc = accuracy_score(y_true_group, y_pred_group)
                    f1 = f1_score(y_true_group, y_pred_group, average='macro')
                    scores[f'fair/ACC/{group}'] = float(acc)
                    scores[f'fair/F1_macro/{group}'] = float(f1)
                else:  # for binary or multilabel
                    prauc = average_precision_score(y_true[mask], y_score[mask])
                    auroc = roc_auc_score(y_true[mask], y_score[mask])
                    scores[f'fair/PRAUC/{group}'] = float(prauc)
                    scores[f'fair/AUROC/{group}'] = float(auroc)
            except ValueError:
                continue
        return scores

    # Lightning inference step for prediction
    def predict_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        return out['predictions'].detach()

    # Optimizer and scheduler configuration (AdamW + ReduceLROnPlateau)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=getattr(self.hparams, 'lr', 0.0001)
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=getattr(self.hparams, 'patience', 10),
                mode='min',
                verbose=True
            ),
            "monitor": "loss/validation_epoch",
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # Print model statistics at fit start (param count, model size)
    def on_fit_start(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = sum(p.element_size() * p.numel() for p in self.parameters()) / 1024 ** 2
        if self.logger and hasattr(self.logger, 'experiment'):
            exp = self.logger.experiment
            if hasattr(exp, 'add_scalar'):
                exp.add_scalar("model/total_params", total_params, 0)
                exp.add_scalar("model/trainable_params", trainable_params, 0)
                exp.add_scalar("model/size_MB", model_size_mb, 0)
        print(f"[Model Summary] Total params: {total_params}, Trainable: {trainable_params}, Size: {model_size_mb:.2f} MB")
        self.epoch_start_time = time.time()

    def on_train_epoch_start(self):
        # Reset peak memory stats & record epoch start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.total_time += epoch_time
        self.log("system/epoch_time", epoch_time, sync_dist=True)
        self.log("system/total_train_time", self.total_time, sync_dist=True)
        self.log_gpu_usage()
        # Optionally save features after training epoch
        if getattr(self.hparams, 'save', False):
            self.feature_saver.save_features('train', self.current_epoch, getattr(self.hparams, 'hidden_size', 256))

    # Log GPU memory usage statistics
    def log_gpu_usage(self):
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
            self.log("system/max_vram_MB", mem_allocated, sync_dist=True)
            torch.cuda.reset_peak_memory_stats()
    
    def on_fit_end(self):
        # Log total training time at the end of fitting
        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_scalar("system/total_training_seconds", self.total_time, self.current_epoch)
        print(f"[Training Completed] Total training time: {self.total_time:.2f}s")

    # Initialize the feature saver utility (if enabled)
    def _init_feature_storage(self):
        if getattr(self.hparams, 'save', False):
            self.feature_saver = FeatureSaver(
                save_dir="./features",
                task=self.hparams.task,
                model_name=self.hparams['model_name'],
                seed=self.hparams.seed
            )
        
    # Unified loss computation method depending on task type and label weights
    def classification_loss(self, logits, labels):
        if self.hparams.task == 'phenotype':  
            if hasattr(self.hparams, 'label_weights') and self.hparams.label_weights is not None:
                weights = torch.tensor(self.hparams.label_weights, dtype=torch.float32, device=logits.device)
                if weights.dim() == 1:
                    weights = weights.unsqueeze(0).expand(logits.shape[0], -1)
                loss = F.binary_cross_entropy_with_logits(
                    logits, 
                    labels,
                    reduction='none'
                )
                loss = (loss * weights).mean()
                return loss
            else:
                return F.binary_cross_entropy_with_logits(
                    logits, 
                    labels,
                    reduction='mean',
                    pos_weight=getattr(self.hparams, 'pos_weight', None)
                )
        elif self.hparams.task == 'mortality':  
            if hasattr(self.hparams, 'label_weights') and self.hparams.label_weights is not None:
                pos_weight = torch.tensor(self.hparams.label_weights[1], dtype=torch.float32, device=logits.device)
                return F.binary_cross_entropy_with_logits(
                    logits.view(-1), 
                    labels.float().view(-1),
                    reduction='mean',
                    pos_weight=pos_weight
                )
            else:
                return F.binary_cross_entropy_with_logits(
                    logits.view(-1), 
                    labels.float().view(-1),
                    reduction='mean',
                    pos_weight=getattr(self.hparams, 'mortality_pos_weight', None)
                )
        elif self.hparams.task == 'los':  
            if hasattr(self.hparams, 'label_weights') and self.hparams.label_weights is not None:
                weights = torch.tensor(self.hparams.label_weights, dtype=torch.float32, device=logits.device)
                return F.cross_entropy(
                    logits, 
                    labels.long(),
                    weight=weights,
                    reduction='mean'
                )
            else:
                return F.cross_entropy(
                    logits, 
                    labels.long().squeeze(),
                    weight=getattr(self.hparams, 'class_weight', None),
                    reduction='mean'
                )
        else:  
            if hasattr(self.hparams, 'label_weights') and self.hparams.label_weights is not None:
                weights = torch.tensor(self.hparams.label_weights, dtype=torch.float32, device=logits.device)
                return F.cross_entropy(
                    logits, 
                    labels.long(),
                    weight=weights,
                    reduction='mean'
                )
            else:
                return F.cross_entropy(
                    logits, 
                    labels.long(),
                    weight=getattr(self.hparams, 'class_weight', None),
                    reduction='mean'
                )