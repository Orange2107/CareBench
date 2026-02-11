import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List

from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from ..base.base_encoder import create_ehr_encoder, create_cxr_encoder


@ModelRegistry.register('inforeg')
class InfoReg(BaseFuseTrainer):
    """
    Lightning re-implementation of InfoReg.
    
    Information Regularization for Multimodal Learning.
    - Uses modality-specific and joint classifiers
    - Applies InfoReg penalty to prevent dominant modality overfitting
    - Tracks Fisher Information Matrix (FIM) to adjust regularization
    - Manually optimized training with custom gradient manipulation
    
    Reference: InfoReg - Information Regularization for Multi-Task Learning
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.class_names = getattr(self.hparams, 'class_names', None)
        self.automatic_optimization = False

        task = getattr(self.hparams, 'task', 'phenotype')
        if task == 'phenotype':
            self.num_classes = getattr(self.hparams, 'num_classes', 25)
        elif task == 'mortality':
            self.num_classes = 1
        elif task == 'los':
            self.num_classes = 7  # LoS has 7 classes
        else:
            raise ValueError(f"Unsupported task: {task}. Only 'mortality', 'phenotype', and 'los' are supported")
        
        if task == 'los':
            self.criterion = nn.CrossEntropyLoss()
            self.use_bce = False
        else:
            self.pred_criterion = nn.BCELoss(reduction='none')
            self.use_bce = True

        self.info_k_threshold = getattr(self.hparams, 'info_k_threshold', 0.05)
        self.info_ehr_scale = getattr(self.hparams, 'info_ehr_scale', 0.95)
        self.info_cxr_scale = getattr(self.hparams, 'info_cxr_scale', 0.10)
        self.info_history = getattr(self.hparams, 'info_history', 10)
        
        self.ehr_trace_history: List[float] = []
        self.cxr_trace_history: List[float] = []

        self._build_encoders()
        
        hidden = self.hparams['hidden_size']
        fusion_in_dim = hidden * 2
        self.head_joint = nn.Linear(fusion_in_dim, self.num_classes)
        self.head_ehr = nn.Linear(hidden, self.num_classes)
        self.head_cxr = nn.Linear(hidden, self.num_classes)

        self.ehr_prefixes = ['ehr_model']
        self.cxr_prefixes = ['cxr_model']
        
        self.global_state: Dict[str, torch.Tensor] = {}

    # ---------------------------
    # Model construction helpers
    # ---------------------------
    def _build_encoders(self):
        """Build EHR and CXR encoders using factory functions from base_encoder"""
        hidden = self.hparams['hidden_size']
        
        ehr_encoder_type = getattr(self.hparams, 'ehr_encoder', 'transformer').lower()
        cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50').lower()
        pretrained = getattr(self.hparams, 'pretrained', True)
        
        input_size = getattr(self.hparams, 'input_dim', 49)  
        ehr_kwargs = {
            'input_size': input_size,
            'num_classes': self.num_classes,
            'dropout': getattr(self.hparams, 'ehr_dropout', 0.3),
            'num_layers': getattr(self.hparams, 'ehr_num_layers', 2),
            'n_head': getattr(self.hparams, 'ehr_n_head', 4),
            'bidirectional': getattr(self.hparams, 'ehr_bidirectional', True)
        }
        if ehr_encoder_type == 'transformer':
            ehr_kwargs['d_model'] = hidden  # Transformer uses d_model
        else:
            ehr_kwargs['hidden_size'] = hidden  # LSTM uses hidden_size
        
        self.ehr_model = create_ehr_encoder(
            encoder_type=ehr_encoder_type,
            **ehr_kwargs
        )
        
        self.cxr_model = create_cxr_encoder(
            encoder_type=cxr_encoder_type,
            hidden_size=hidden,
            pretrained=pretrained
        )
        
        print(f"InfoReg initialized with {ehr_encoder_type.upper()} (EHR) + {cxr_encoder_type.upper()} (CXR) for {self.hparams.task} task")

    # ---------------------------
    # Forward pass
    # ---------------------------
    def forward(self, data_dict):
        """Forward pass through EHR and CXR encoders, then through classifier heads"""
        x = data_dict['ehr_ts']
        seq_lengths = data_dict['seq_len']
        img = data_dict['cxr_imgs']

        feat_ehr, _ = self.ehr_model(x, seq_lengths, output_prob=False)
        feat_cxr = self.cxr_model(img)

        feat_fusion = torch.cat((feat_ehr, feat_cxr), dim=1)
        
        logits_joint = self.head_joint(feat_fusion)
        logits_ehr = self.head_ehr(feat_ehr)
        logits_cxr = self.head_cxr(feat_cxr)

        if self.use_bce:
            pred_final = torch.sigmoid(logits_joint)
            pred_ehr = torch.sigmoid(logits_ehr)
            pred_cxr = torch.sigmoid(logits_cxr)
        else:
            pred_final = logits_joint
            pred_ehr = logits_ehr
            pred_cxr = logits_cxr

        outputs = {
            'feat_ehr_distinct': feat_ehr,
            'feat_cxr_distinct': feat_cxr,
            'predictions': pred_final,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
        }
        return outputs

    # ---------------------------
    # Loss computation helpers
    # ---------------------------
    def _compute_masked_loss(self, preds, labels):
        """Compute loss based on task type"""
        if self.use_bce:
            return self.pred_criterion(preds, labels).mean()
        else:
            is_one_hot = (labels.dim() == 2 and labels.shape[1] == self.num_classes and 
                         torch.all((labels == 0) | (labels == 1)))
            
            if is_one_hot:
                labels_indices = labels.argmax(dim=1).long()
            else:
                labels_indices = labels.squeeze().long()
            return self.criterion(preds, labels_indices)

    def _compute_scores(self, preds, labels):
        """
        Compute alignment scores between predictions and labels.
        Used for InfoReg to determine which modality performs better.
        """
        if self.use_bce:
            probs = preds.clamp(1e-6, 1 - 1e-6)
            if labels.dim() == probs.dim():
                alignment = labels * probs + (1 - labels) * (1 - probs)
                return alignment.sum()
            flat_labels = labels.view(-1)
            flat_probs = probs.view(-1)
            aligned = torch.where(flat_labels > 0.5, flat_probs, 1 - flat_probs)
            return aligned.sum()
        else:
            is_one_hot = (labels.dim() == 2 and labels.shape[1] == self.num_classes and 
                         torch.all((labels == 0) | (labels == 1)))
            
            probs = torch.softmax(preds, dim=1)
            if is_one_hot:
                labels_indices = labels.argmax(dim=1).long()
            else:
                labels_indices = labels.squeeze().long()
            gathered = probs.gather(1, labels_indices.view(-1, 1))
            return gathered.sum()

    # ---------------------------
    # InfoReg helpers
    # ---------------------------
    def _is_ehr_param(self, name: str) -> bool:
        """Check if a parameter belongs to EHR encoder"""
        return any(name.startswith(prefix) for prefix in self.ehr_prefixes)

    def _is_cxr_param(self, name: str) -> bool:
        """Check if a parameter belongs to CXR encoder"""
        return any(name.startswith(prefix) for prefix in self.cxr_prefixes)

    def _capture_global_state(self):
        """Keep empty for backward compatibility"""
        pass

    def _apply_inforeg(self, beta_ehr: float, beta_cxr: float):
        """Apply InfoReg regulation to gradients by scaling down dominant modality's gradients"""
        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            if self._is_ehr_param(name) and beta_ehr > 0:
                param.grad *= (1.0 - beta_ehr)
            elif self._is_cxr_param(name) and beta_cxr > 0:
                param.grad *= (1.0 - beta_cxr)

    def _collect_grad_stats(self):
        """Collect gradient statistics for logging"""
        avg_ehr, avg_cxr = 0.0, 0.0
        count_ehr, count_cxr = 0, 0
        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if self._is_ehr_param(name):
                avg_ehr += grad.mean().item()
                count_ehr += 1
            elif self._is_cxr_param(name):
                avg_cxr += grad.mean().item()
                count_cxr += 1
        if count_ehr > 0:
            avg_ehr /= count_ehr
        if count_cxr > 0:
            avg_cxr /= count_cxr
        return avg_ehr, avg_cxr

    def _compute_fim(self):
        """
        Compute Fisher Information Matrix (FIM) approximation.
        FIM is approximated by the squared gradients.
        """
        fim_total = 0.0
        fim_ehr = 0.0
        fim_cxr = 0.0
        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            value = param.grad.detach().pow(2).mean().item()
            fim_total += value
            if self._is_ehr_param(name):
                fim_ehr += value
            elif self._is_cxr_param(name):
                fim_cxr += value
        return fim_total, fim_ehr, fim_cxr

    def _update_history(self, history: List[float], value: float):
        """Update history buffer with new value"""
        history.append(value)
        max_len = max(self.info_history + 2, 2)
        if len(history) > max_len:
            history.pop(0)

    def _compute_k_value(self):
        """
        Compute k value for InfoReg.
        k measures the change in FIM trace over recent history.
        High k means the model is still learning, low k means convergence.
        """
        if len(self.ehr_trace_history) < self.info_history + 1:
            return 1.0
        recent = self.ehr_trace_history[-self.info_history:]
        previous = self.ehr_trace_history[-(self.info_history + 1):-1]
        tr1 = sum(recent) / len(recent) if recent else 0.0
        tr2 = sum(previous) / len(previous) if previous else 0.0
        if tr1 == 0:
            return 0.0
        return (tr1 - tr2) / tr1

    # ---------------------------
    # Training step with InfoReg
    # ---------------------------
    def training_step(self, batch, batch_idx):
        """
        Training step with InfoReg logic.
        
        InfoReg procedure:
        1. Compute losses for joint, EHR-only, and CXR-only predictions
        2. Perform single backward pass
        3. Compute modality scores to determine dominance
        4. Apply InfoReg: scale down dominant modality's gradients
        5. Update Fisher Information Matrix history
        6. Optimizer step
        """
        opt = self.optimizers()
        opt.zero_grad()
        
        out = self.forward(batch)
        labels = batch['labels']

        loss_joint = self._compute_masked_loss(out['predictions'], labels)
        loss_ehr = self._compute_masked_loss(out['pred_ehr'], labels)
        loss_cxr = self._compute_masked_loss(out['pred_cxr'], labels)
        total_loss = loss_joint + loss_ehr + loss_cxr

        with torch.no_grad():
            score_ehr = self._compute_scores(out['pred_ehr'], labels)
            score_cxr = self._compute_scores(out['pred_cxr'], labels)
        
        total_loss.backward()

        k_value = self._compute_k_value()
        
        beta_ehr = 0.0
        beta_cxr = 0.0
        if k_value > self.info_k_threshold:
            if score_ehr > score_cxr:
                gap = (score_ehr - score_cxr) / (score_ehr + score_cxr + 1e-8)
                beta_ehr = min(self.info_ehr_scale * gap.item(), 0.9)
            elif score_cxr > score_ehr:
                gap = (score_cxr - score_ehr) / (score_ehr + score_cxr + 1e-8)
                beta_cxr = min(self.info_cxr_scale * gap.item(), 0.9)

        self._apply_inforeg(beta_ehr, beta_cxr)
        
        fim_total, fim_ehr, fim_cxr = self._compute_fim()
        self._update_history(self.ehr_trace_history, fim_ehr)
        self._update_history(self.cxr_trace_history, fim_cxr)

        opt.step()

        self.log_dict(
            {
                'train/loss': total_loss.detach(),
                'loss/train_joint': loss_joint.detach(),
                'loss/train_ehr': loss_ehr.detach(),
                'loss/train_cxr': loss_cxr.detach(),
                'fim/total': fim_total,
                'fim/ehr': fim_ehr,
                'fim/cxr': fim_cxr,
                'info/k_value': k_value,
                'info/beta_ehr': beta_ehr,
                'info/beta_cxr': beta_cxr,
                'info/score_ehr': score_ehr.item(),
                'info/score_cxr': score_cxr.item(),
            },
            on_step=True,
            on_epoch=True,
            batch_size=labels.size(0),
            sync_dist=True
        )
        
        return {'loss': total_loss.detach()}

    # ---------------------------
    # Validation / Test steps
    # ---------------------------
    def _val_test_shared_step(self, batch, cache):
        """Shared logic for validation and test steps"""
        out = self._shared_step(batch)
        cache['predictions'].append(out['predictions'].detach())
        cache['pred_ehr'].append(out['pred_ehr'].detach())
        cache['pred_cxr'].append(out['pred_cxr'].detach())
        cache['labels'].append(batch['labels'].detach())

        if 'meta_attrs' in batch:
            if 'meta_attrs' not in cache:
                cache['meta_attrs'] = []
            meta_attrs_list = batch['meta_attrs'].to_dict('records')
            cache['meta_attrs'].extend(meta_attrs_list)

        if 'groups' in batch:
            if 'groups' not in cache:
                cache['groups'] = []
            cache['groups'].extend(batch['groups'])

        return out

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        out = self._val_test_shared_step(batch, self.val_info)
        loss_total = self._compute_masked_loss(out['predictions'], batch['labels'])
        self.log_dict(
            {'loss/validation': loss_total.detach()},
            on_step=True,
            on_epoch=True,
            batch_size=batch['labels'].shape[0],
            sync_dist=True
        )
        return loss_total

    def test_step(self, batch, batch_idx):
        """Test step"""
        self._val_test_shared_step(batch, self.test_info)

    # ---------------------------
    # Epoch hooks
    # ---------------------------
    def on_train_epoch_start(self):
        """Capture global state at the start of each training epoch"""
        self._capture_global_state()

    def _get_ehr_cxr_scores(self, info_cache, clear_cache=False):
        """
        Compute separate evaluation scores for EHR and CXR modalities.
        This helps analyze individual modality performance.
        """
        if not info_cache['pred_ehr'] or not info_cache['pred_cxr']:
            return {}, {}
        
        preds_ehr = torch.cat(info_cache['pred_ehr'])
        preds_cxr = torch.cat(info_cache['pred_cxr'])
        labels = torch.cat(info_cache['labels'])
        
        meta_attrs = None
        if info_cache.get('meta_attrs'):
            import pandas as pd
            meta_attrs = pd.DataFrame(info_cache['meta_attrs'])
        
        scores_ehr = self.evaluate_performance(preds_ehr, labels, meta_attrs=meta_attrs)
        scores_cxr = self.evaluate_performance(preds_cxr, labels, meta_attrs=meta_attrs)
        
        return scores_ehr, scores_cxr

    def on_validation_epoch_end(self):
        """Compute and log validation metrics including per-modality scores"""
        scores_ehr, scores_cxr = self._get_ehr_cxr_scores(self.val_info, clear_cache=False)
        scores = self._val_test_epoch_end(self.val_info, clear_cache=True)
        
        combined = {
            **scores,
            **{f'ehr_{k}': v for k, v in scores_ehr.items()},
            **{f'cxr_{k}': v for k, v in scores_cxr.items()},
        }
        combined['step'] = float(self.current_epoch)
        
        self.log_dict({k: v for k, v in scores.items() if not isinstance(v, (list, str))}, 
                      on_epoch=True, 
                      on_step=False, 
                      sync_dist=True)
        return scores

    def on_test_epoch_end(self):
        """Compute and store test metrics including per-modality scores"""
        if getattr(self.hparams, 'save_predictions', False):
            self._save_test_predictions_and_labels()
        
        scores = self._val_test_epoch_end(self.test_info, clear_cache=False)
        scores_ehr, scores_cxr = self._get_ehr_cxr_scores(self.test_info, clear_cache=True)
        
        combined = {
            **scores,
            **{f'ehr_{k}': v for k, v in scores_ehr.items()},
            **{f'cxr_{k}': v for k, v in scores_cxr.items()},
        }
        self.test_results = combined
    
    # ---------------------------
    # Optimizer configuration
    # ---------------------------
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
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

