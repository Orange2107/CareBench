import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from ..base.base_encoder import LSTMEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm

@ModelRegistry.register('lstm')
class LSTMModel(BaseFuseTrainer):
    """
    Single-modal LSTM model for EHR data using MedFuse-style LSTMEncoder
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.task = self.hparams.task
        
        if self.task == 'phenotype':
            self.num_classes = self.hparams.num_classes
        elif self.task == 'mortality':
            self.num_classes = 1
        elif self.task == 'los':
            self.num_classes = 7
        else:
            raise ValueError(f"Unsupported task: {self.task}. Only 'mortality', 'phenotype', and 'los' are supported")
        
        self._init_model_components()

    def _init_model_components(self):
        self.lstm_encoder = LSTMEncoder(
            input_size=self.hparams.input_dim,
            num_classes=self.num_classes,
            hidden_size=getattr(self.hparams, 'hidden_size', 256),
            num_layers=getattr(self.hparams, 'num_layers', 2),
            dropout=getattr(self.hparams, 'dropout', 0.3),
            bidirectional=getattr(self.hparams, 'bidirectional', True)
        )
        
        print(f"LSTM model initialized for {self.task} task")
        print(f"  - Input dimension: {self.hparams.input_dim}")
        print(f"  - Hidden size: {getattr(self.hparams, 'hidden_size', 256)}")
        print(f"  - Number of layers: {getattr(self.hparams, 'num_layers', 2)}")
        print(f"  - Bidirectional: {getattr(self.hparams, 'bidirectional', True)}")
        print(f"  - Dropout: {getattr(self.hparams, 'dropout', 0.3)}")
        print(f"  - Number of classes: {self.num_classes}")

    def forward(self, batch):
        x = batch['ehr_ts']
        seq_lengths = batch['seq_len']
        labels = batch['labels']
        features, predictions = self.lstm_encoder(x, seq_lengths, output_prob=False)
        loss = self.classification_loss(predictions, labels)
        return {
            'loss': loss,
            'predictions': torch.sigmoid(predictions),
            'labels': labels,
            'features': features
        }

    def get_permutation_importance(self, dataloader, feature_names=None, n_repeats=5, metric='auroc'):
        print(f"Computing permutation importance with metric: {metric}")
        self.eval()
        device = next(self.parameters()).device
        print(f"Model device: {device}")
        def compute_metric(predictions, labels, metric_name):
            predictions_np = predictions.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            if metric_name == 'accuracy':
                if self.task == 'mortality':
                    return ((predictions_np > 0.5) == (labels_np > 0.5)).mean()
                else:
                    return ((predictions_np > 0.5) == (labels_np > 0.5)).mean()
            elif metric_name == 'auroc':
                from sklearn.metrics import roc_auc_score
                try:
                    if self.task == 'mortality':
                        return roc_auc_score(labels_np, predictions_np)
                    else:
                        return roc_auc_score(labels_np, predictions_np, average='macro')
                except:
                    return 0.0
            elif metric_name == 'f1':
                from sklearn.metrics import f1_score
                if self.task == 'mortality':
                    preds_binary = (predictions_np > 0.5).astype(int)
                    return f1_score(labels_np, preds_binary)
                else:
                    preds_binary = (predictions_np > 0.5).astype(int)
                    return f1_score(labels_np, preds_binary, average='macro')
            elif metric_name == 'prauc':
                from sklearn.metrics import average_precision_score
                try:
                    if self.task == 'mortality':
                        return average_precision_score(labels_np, predictions_np)
                    else:
                        return average_precision_score(labels_np, predictions_np, average='macro')
                except:
                    return 0.0
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
        
        with torch.no_grad():
            print("Computing baseline performance...")
            baseline_scores = []
            for batch in tqdm(dataloader, desc="Baseline evaluation"):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                out = self(batch)
                score = compute_metric(out['predictions'], out['labels'], metric)
                baseline_scores.append(score)
            baseline_score = np.mean(baseline_scores)
            print(f"Baseline {metric}: {baseline_score:.4f}")
            sample_batch = next(iter(dataloader))
            sample_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample_batch.items()}
            n_features = sample_batch['ehr_ts'].shape[-1]
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(n_features)]
            print(f"Computing permutation importance for {n_features} features...")
            feature_importance_scores = []
            feature_importance_stds = []
            for feature_idx in tqdm(range(n_features), desc="Computing feature importance"):
                scores_with_permutation = []
                for repeat in range(n_repeats):
                    repeat_scores = []
                    for batch in dataloader:
                        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                        perturbed_batch = {}
                        for k, v in batch.items():
                            if torch.is_tensor(v):
                                perturbed_batch[k] = v.clone()
                            else:
                                perturbed_batch[k] = v
                        feature_data = perturbed_batch['ehr_ts'][:, :, feature_idx].clone()
                        batch_size, seq_len = feature_data.shape
                        feature_flat = feature_data.flatten()
                        perm_indices = torch.randperm(feature_flat.numel(), device=device)
                        feature_permuted = feature_flat[perm_indices].reshape(batch_size, seq_len)
                        perturbed_batch['ehr_ts'][:, :, feature_idx] = feature_permuted
                        out = self(perturbed_batch)
                        score = compute_metric(out['predictions'], out['labels'], metric)
                        repeat_scores.append(score)
                    scores_with_permutation.append(np.mean(repeat_scores))
                importance_scores = [baseline_score - score for score in scores_with_permutation]
                feature_importance_scores.append(np.mean(importance_scores))
                feature_importance_stds.append(np.std(importance_scores))
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': feature_importance_scores,
            'importance_std': feature_importance_stds,
            'importance_abs_mean': np.abs(feature_importance_scores)
        }).sort_values('importance_abs_mean', ascending=False)
        
        print(f"\nTop 10 most important features:")
        print(importance_df.head(10))
        
        return importance_df

    def save_feature_importance(self, importance_df, save_path):
        csv_path = save_path.replace('.png', '.csv')
        importance_df.to_csv(csv_path, index=False)
        print(f"Feature importance saved to: {csv_path}")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        colors = ['blue' if x >= 0 else 'red' for x in top_features['importance_mean']]
        bars = plt.barh(range(len(top_features)), top_features['importance_mean'], color=colors, alpha=0.7)
        plt.errorbar(top_features['importance_mean'], range(len(top_features)), 
                    xerr=top_features['importance_std'], fmt='none', color='black', alpha=0.5)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance Score')
        plt.title('LSTM Feature Importance (Permutation Method)')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Feature importance plot saved to: {save_path}")

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
