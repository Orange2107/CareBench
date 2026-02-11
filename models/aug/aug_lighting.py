import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from ..base.base_encoder import create_ehr_encoder, create_cxr_encoder


@ModelRegistry.register('aug')
class AUG(BaseFuseTrainer):
    """
    Lightning re-implementation of AUG (Adaptive classifier assignment + sustained boosting).
    Mirrors the original training logic while using the project's configurable encoders.
    
    Reference: Adaptive Classifier Assignment + Sustained Boosting
    - Two modality-specific classifiers trained separately with manual optimization
    - Adaptive Classifier Assignment (ACA) based on modality performance ratio
    - Dynamic layer addition for sustained boosting
    - Uses configurable encoders from base_encoder (LSTM/Transformer for EHR, ResNet/Swin for CXR)
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
            self.num_classes = 7  # LoS has 7 classes (bins 2-8, excluding 0,1)
        else:
            raise ValueError(f"Unsupported task: {task}. Only 'mortality', 'phenotype', and 'los' are supported")
        
        if task == 'los':
            self.criterion = nn.CrossEntropyLoss()
            self.use_bce = False
        else:
            reduction = 'none'
            self.pred_criterion = nn.BCEWithLogitsLoss(reduction=reduction)
            self.use_bce = True
            self.bce_loss = nn.BCEWithLogitsLoss()

        self.merge_alpha = getattr(self.hparams, 'aug_merge_alpha', 0.4)
        self.aug_lambda = getattr(self.hparams, 'aug_lambda', 1.0)
        self.aug_margin = getattr(self.hparams, 'aug_margin', 0.01)
        interval = getattr(self.hparams, 'aug_layer_check_interval', 10)
        self.aug_layer_check_interval = max(1, int(interval))
        self.classifier_hidden_dim = getattr(self.hparams, 'aug_classifier_hidden_dim', 256)

        self.train_score_ehr = 0.0
        self.train_score_cxr = 0.0

        self._build_encoders()
        self._build_classifiers()

    # ---------------------------
    # Model construction helpers
    # ---------------------------
    def _build_encoders(self):
        """Build encoders using factory functions from base_encoder"""
        hidden = self.hparams['hidden_size']
        
        ehr_encoder_type = getattr(self.hparams, 'ehr_encoder', 'transformer').lower()
        cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50').lower()
        pretrained = getattr(self.hparams, 'pretrained', True)
        
        input_size = getattr(self.hparams, 'input_dim', 49)  
        ehr_params = {
            'input_size': input_size,
            'num_classes': self.num_classes,
            'hidden_size': hidden,
            'dropout': getattr(self.hparams, 'ehr_dropout', 0.3),
        }
        
        if ehr_encoder_type == 'lstm':
            ehr_params.update({
                'num_layers': getattr(self.hparams, 'ehr_num_layers', 2),
                'bidirectional': getattr(self.hparams, 'ehr_bidirectional', True)
            })
        elif ehr_encoder_type == 'transformer':
            ehr_params.update({
                'd_model': ehr_params.pop('hidden_size'),  # transformer uses d_model
                'n_head': getattr(self.hparams, 'ehr_n_head', 4),
                'n_layers': getattr(self.hparams, 'ehr_n_layers', 1),
            })
        
        self.ehr_model = create_ehr_encoder(encoder_type=ehr_encoder_type, **ehr_params)
        
        cxr_params = {
            'hidden_size': hidden,
            'pretrained': pretrained
        }
        
        self.cxr_model = create_cxr_encoder(encoder_type=cxr_encoder_type, **cxr_params)
        
        self.modality_dim_ehr = hidden
        self.modality_dim_cxr = hidden
        
        print(f"AUG model encoders initialized:")
        print(f"  - EHR encoder: {ehr_encoder_type}, hidden_dim: {hidden}")
        print(f"  - CXR encoder: {cxr_encoder_type}, hidden_dim: {hidden}, pretrained: {pretrained}")

    def _build_classifiers(self):
        """Build modality-specific classifiers for AUG"""
        self.embedding_ehr = nn.Sequential(
            nn.Linear(self.modality_dim_ehr, self.classifier_hidden_dim),
            nn.ReLU(),
        )
        self.embedding_cxr = nn.Sequential(
            nn.Linear(self.modality_dim_cxr, self.classifier_hidden_dim),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(self.classifier_hidden_dim, self.num_classes)
        self.additional_layers_ehr = nn.ModuleList()
        self.additional_layers_cxr = nn.ModuleList()
        self.relu = nn.ReLU()

    # ---------------------------
    # Feature extraction helpers
    # ---------------------------
    def _extract_features(self, data_dict):
        """Extract features from clinical dataset (EHR time series + CXR images)"""
        x = data_dict['ehr_ts']
        seq_lengths = data_dict['seq_len']
        img = data_dict['cxr_imgs']
        
        feat_ehr, _ = self.ehr_model(x, seq_lengths, output_prob=False)
        feat_cxr = self.cxr_model(img)
        
        return feat_ehr, feat_cxr

    def _classify(self, features, is_ehr):
        """Classify with optional additional layers for sustained boosting"""
        if is_ehr:
            embed = self.embedding_ehr(features)
            logits = self.fc_out(embed)
            prev_logits = logits
            add_logits = None
            layers = self.additional_layers_ehr
        else:
            embed = self.embedding_cxr(features)
            logits = self.fc_out(embed)
            prev_logits = logits
            add_logits = None
            layers = self.additional_layers_cxr

        if layers:
            for idx, layer in enumerate(layers):
                add_feat = self.relu(layer(features))
                add_logits = self.fc_out(add_feat)
                logits = logits + add_logits
                if idx < len(layers) - 1:
                    prev_logits = logits

        return logits, prev_logits, add_logits

    def add_layer(self, is_ehr=True):
        """Dynamically add a layer for sustained boosting"""
        in_dim = self.modality_dim_ehr if is_ehr else self.modality_dim_cxr
        new_layer = nn.Linear(in_dim, self.classifier_hidden_dim)
        nn.init.xavier_normal_(new_layer.weight)
        nn.init.constant_(new_layer.bias, 0)
        new_layer = new_layer.to(self.device)
        if is_ehr:
            self.additional_layers_ehr.append(new_layer)
        else:
            self.additional_layers_cxr.append(new_layer)

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict to handle dynamic additional layers.
        AUG model can have different numbers of additional layers depending on training dynamics.
        This method ensures the model architecture matches the checkpoint before loading.
        """
        ehr_layer_keys = [k for k in state_dict.keys() if k.startswith('additional_layers_ehr.')]
        cxr_layer_keys = [k for k in state_dict.keys() if k.startswith('additional_layers_cxr.')]
        
        # Each layer has 2 keys (weight and bias), so divide by 2 to get layer count
        ehr_layer_count = len(set(k.split('.')[1] for k in ehr_layer_keys if len(k.split('.')) > 1))
        cxr_layer_count = len(set(k.split('.')[1] for k in cxr_layer_keys if len(k.split('.')) > 1))
        
        current_ehr_layers = len(self.additional_layers_ehr)
        current_cxr_layers = len(self.additional_layers_cxr)
        
        for _ in range(ehr_layer_count - current_ehr_layers):
            self.add_layer(is_ehr=True)
        
        for _ in range(cxr_layer_count - current_cxr_layers):
            self.add_layer(is_ehr=False)
        
        return super().load_state_dict(state_dict, strict=False)

    # ---------------------------
    # Forward / inference logic
    # ---------------------------
    def forward(self, data_dict):
        """
        Forward pass for prediction and loss calculation.
        Returns dict with loss and labels for consistency with project.
        """
        feat_ehr, feat_cxr = self._extract_features(data_dict)
        logits_ehr, _, _ = self._classify(feat_ehr, is_ehr=True)
        logits_cxr, _, _ = self._classify(feat_cxr, is_ehr=False)
        
        fused_logits = self.merge_alpha * logits_ehr + (1 - self.merge_alpha) * logits_cxr

        if self.use_bce:
            pred_final = torch.sigmoid(fused_logits)
            pred_ehr = torch.sigmoid(logits_ehr)
            pred_cxr = torch.sigmoid(logits_cxr)
        else:
            pred_final = fused_logits
            pred_ehr = logits_ehr
            pred_cxr = logits_cxr

        labels = data_dict['labels']
        loss = self._prediction_loss(pred_final, labels)

        outputs = {
            'feat_ehr_distinct': feat_ehr,
            'feat_cxr_distinct': feat_cxr,
            'predictions': pred_final,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
            'loss': loss,
            'labels': labels,
        }
        return outputs

    # ---------------------------
    # Loss helpers
    # ---------------------------
    def _prepare_one_hot(self, labels):
        """Prepare one-hot labels for CE loss"""
        if labels.dim() == 1:
            return F.one_hot(labels.long(), num_classes=self.num_classes).float()
        return labels.float()

    def _compute_ce_loss(self, logits, prev_logits, add_logits, labels):
        """CE loss with KL regularization for sustained boosting"""
        is_one_hot = (labels.dim() == 2 and labels.shape[1] == self.num_classes and 
                     torch.all((labels == 0) | (labels == 1)))
        
        if is_one_hot:
            labels_indices = labels.argmax(dim=1).long()
            labels_one_hot = labels.float()
        else:
            labels_indices = labels.squeeze().long()
            labels_one_hot = F.one_hot(labels_indices, num_classes=self.num_classes).float()
        
        if add_logits is None:
            return self.criterion(logits, labels_indices)
        
        add_log_probs = F.log_softmax(add_logits, dim=1)
        prev_probs = torch.softmax(prev_logits.detach(), dim=1)
        kl = F.kl_div(add_log_probs, prev_probs, reduction='batchmean')
        
        loss = (
            self.criterion(logits, labels_indices)
            + self.criterion(prev_logits, labels_indices)
            + self.criterion(add_logits, labels_indices)
            - 0.5 * kl
        )
        return loss

    def _compute_bce_loss(self, logits, prev_logits, add_logits, labels):
        """BCE loss with KL regularization for sustained boosting"""
        if add_logits is None:
            return self.bce_loss(logits, labels.float())
        kl = labels.float() * torch.sigmoid(prev_logits.detach())
        loss = (
            self.bce_loss(logits, labels.float())
            + self.bce_loss(prev_logits, labels.float())
            + self.bce_loss(add_logits, labels.float())
            - 0.5 * self.bce_loss(add_logits, kl)
        )
        return loss

    def _prediction_loss(self, preds, labels):
        """
        Prediction loss calculation logic.
        For BCE: preds are probabilities (after sigmoid), need to convert back to logits.
        For CE: preds are logits, use directly.
        """
        if self.use_bce:
            if labels.dtype != preds.dtype:
                labels = labels.float()
            return self.pred_criterion(torch.logit(preds.clamp(1e-6, 1 - 1e-6)), labels).mean()
        else:
            is_one_hot = (labels.dim() == 2 and labels.shape[1] == self.num_classes and 
                         torch.all((labels == 0) | (labels == 1)))
            
            if is_one_hot:
                labels_indices = labels.argmax(dim=1).long()
            else:
                labels_indices = labels.squeeze().long()
            return self.criterion(preds, labels_indices)

    def _score_modality(self, logits, labels):
        """Score modality performance for ACA"""
        if self.use_bce:
            probs = torch.sigmoid(logits.detach())
            if labels.dim() == probs.dim():
                aligned = labels * probs + (1 - labels) * (1 - probs)
            else:
                labels = labels.view(-1, 1)
                aligned = torch.where(labels > 0.5, probs, 1 - probs)
            return aligned.sum().item()

        if labels.dim() == 1:
            indices = labels.long()
        else:
            indices = labels.argmax(dim=1)
        probs = torch.softmax(logits.detach(), dim=1)
        gathered = probs.gather(1, indices.view(-1, 1))
        return gathered.sum().item()

    # ---------------------------
    # Training logic
    # ---------------------------
    def training_step(self, batch, batch_idx):
        """
        Training step with manual optimization.
        Trains two modality-specific classifiers separately.
        """
        opt = self.optimizers()
        batch = self._BaseFuseTrainer__get_batch_data(batch)
        labels = batch['labels']
        feat_ehr, feat_cxr = self._extract_features(batch)

        logits_ehr, prev_ehr, add_ehr = self._classify(feat_ehr, is_ehr=True)
        if self.use_bce:
            loss_ehr = self._compute_bce_loss(logits_ehr, prev_ehr, add_ehr, labels)
        else:
            loss_ehr = self._compute_ce_loss(logits_ehr, prev_ehr, add_ehr, labels)

        opt.zero_grad()
        self.manual_backward(loss_ehr)
        opt.step()

        logits_cxr, prev_cxr, add_cxr = self._classify(feat_cxr, is_ehr=False)
        if self.use_bce:
            loss_cxr = self._compute_bce_loss(logits_cxr, prev_cxr, add_cxr, labels)
        else:
            loss_cxr = self._compute_ce_loss(logits_cxr, prev_cxr, add_cxr, labels)

        opt.zero_grad()
        self.manual_backward(loss_cxr)
        opt.step()

        total_loss = self.merge_alpha * loss_ehr + (1 - self.merge_alpha) * loss_cxr

        with torch.no_grad():
            self.train_score_ehr += self._score_modality(logits_ehr, labels)
            self.train_score_cxr += self._score_modality(logits_cxr, labels)

        self.log_dict(
            {
                'loss/train': total_loss.detach(),
                'train/loss': total_loss.detach(),
                'loss/train_ehr': loss_ehr.detach(),
                'loss/train_cxr': loss_cxr.detach(),
            },
            on_step=True,
            on_epoch=True,
            batch_size=labels.size(0),
            sync_dist=True,
        )
        
        return {
            'loss': total_loss.detach(),
            'pred': None,
            'labels': labels.detach()
        }

    # ---------------------------
    # Validation / Test overrides
    # ---------------------------
    def validation_step(self, batch, batch_idx):
        """Validation step using base class method with AUG-specific logging"""
        out = self._val_test_shared_step(batch, self.val_info)
        
        if 'pred_ehr' in out:
            self.val_info['pred_ehr'].append(out['pred_ehr'].detach())
        if 'pred_cxr' in out:
            self.val_info['pred_cxr'].append(out['pred_cxr'].detach())
        
        batch_size = out['labels'].shape[0] if 'labels' in out else batch['labels'].shape[0]
        self.log_dict({
            'loss/validation': out['loss'].detach(),
            'loss/validation_epoch': out['loss'].detach(),
        }, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)
        
        return out['loss'].detach()

    def test_step(self, batch, batch_idx):
        """Test step using base class method with AUG-specific evaluation"""
        out = self._val_test_shared_step(batch, self.test_info)
        
        if 'pred_ehr' in out:
            self.test_info['pred_ehr'].append(out['pred_ehr'].detach())
        if 'pred_cxr' in out:
            self.test_info['pred_cxr'].append(out['pred_cxr'].detach())
        
        return out

    # ---------------------------
    # Epoch hooks
    # ---------------------------
    def on_train_epoch_start(self):
        """Reset ACA score trackers"""
        self.train_score_ehr = 0.0
        self.train_score_cxr = 0.0

    def on_train_epoch_end(self):
        """
        Adaptive Classifier Assignment (ACA).
        Dynamically add layers based on modality performance ratio.
        """
        ratio = self.train_score_ehr / (self.train_score_cxr + 1e-8)
        should_check = (self.current_epoch == 0) or ((self.current_epoch + 1) % self.aug_layer_check_interval == 0)
        if should_check:
            if ratio > self.aug_lambda + self.aug_margin:
                self.add_layer(is_ehr=False)
                print(f"[AUG ACA] Epoch {self.current_epoch}: Added layer to CXR classifier (ratio={ratio:.4f})")
            elif ratio < self.aug_lambda - self.aug_margin:
                self.add_layer(is_ehr=True)
                print(f"[AUG ACA] Epoch {self.current_epoch}: Added layer to EHR classifier (ratio={ratio:.4f})")
        
        self.log(
            'aug/ratio_ehr_to_cxr',
            ratio,
            on_step=False,
            on_epoch=True,
        )

    def _get_ehr_cxr_scores(self, cache, clear_cache=False):
        """Compute scores for individual modalities (AUG-specific)"""
        if not cache.get('pred_ehr') or not cache.get('pred_cxr') or not cache.get('labels'):
            return {}, {}
        
        pred_ehr = torch.cat(cache['pred_ehr'])
        pred_cxr = torch.cat(cache['pred_cxr'])
        labels = torch.cat(cache['labels'])
        
        scores_ehr = self.evaluate_performance(pred_ehr, labels)
        scores_cxr = self.evaluate_performance(pred_cxr, labels)
        
        if clear_cache:
            cache['pred_ehr'] = []
            cache['pred_cxr'] = []
        
        return scores_ehr, scores_cxr

    def on_validation_epoch_end(self):
        """Validation epoch end with AUG-specific modality scores"""
        scores_ehr, scores_cxr = self._get_ehr_cxr_scores(self.val_info, clear_cache=False)
        scores = self._val_test_epoch_end(self.val_info, clear_cache=True)
        
        combined = {
            **scores,
            **{f'ehr_{k}': v for k, v in scores_ehr.items()},
            **{f'cxr_{k}': v for k, v in scores_cxr.items()},
        }
        combined['step'] = float(self.current_epoch)
        
        self.log_dict({k: v for k, v in scores.items() if not isinstance(v, (list, str))}, on_epoch=True, on_step=False, sync_dist=True)
        return scores

    def on_test_epoch_end(self):
        """Test epoch end with AUG-specific modality scores"""
        if getattr(self.hparams, 'save_predictions', False):
            self._save_test_predictions_and_labels()
        
        scores = self._val_test_epoch_end(self.test_info, clear_cache=False)
        scores_ehr, scores_cxr = self._get_ehr_cxr_scores(self.test_info, clear_cache=True)
        
        combined = {
            **scores,
            **{f'ehr_{k}': v for k, v in scores_ehr.items()},
            **{f'cxr_{k}': v for k, v in scores_cxr.items()},
        }
        self.test_results = {x: combined[x] for x in combined}

    # ---------------------------
    # Optimizer
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
