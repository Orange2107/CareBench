import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from .mmtm_components import MMTMLayer, KLDivLoss, CosineLoss
from ..base.base_encoder import create_ehr_encoder, create_cxr_encoder
import os

@ModelRegistry.register('mmtm')
class MMTM(BaseFuseTrainer):
    """MMTM-based multimodal fusion model with configurable encoders"""

    @property
    def automatic_optimization(self):
        return False

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.val_info = {'predictions': [], 'labels': [], 'pred_ehr': [], 'pred_cxr': [], 'groups': [], 'meta_attrs': []}
        self.test_info = {'predictions': [], 'labels': [], 'pred_ehr': [], 'pred_cxr': [], 'groups': [], 'meta_attrs': []}
        self.task = self.hparams.task

        if self.task == 'phenotype':
            self.num_classes = self.hparams.num_classes
        elif self.task == 'mortality':
            self.num_classes = 1
        elif self.task == 'los':
            self.num_classes = 7
        else:
            raise ValueError(f"Unsupported task: {self.task}. Only 'mortality', 'phenotype', and 'los' are supported.")

        if not hasattr(self.hparams, 'mmtm_ratio'):
            self.hparams.mmtm_ratio = 4
        if not hasattr(self.hparams, 'layer_after'):
            self.hparams.layer_after = 0
        if not hasattr(self.hparams, 'ehr_encoder'):
            self.hparams.ehr_encoder = 'lstm'
        if not hasattr(self.hparams, 'cxr_encoder'):
            self.hparams.cxr_encoder = 'resnet50'
        if not hasattr(self.hparams, 'pretrained'):
            self.hparams.pretrained = True

        self._init_model_components()
        if hasattr(self.hparams, 'load_state') and self.hparams.load_state:
            self.load_state()

    def _init_model_components(self):
        ehr_encoder_type = getattr(self.hparams, 'ehr_encoder', 'lstm').lower()
        cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50').lower()
        pretrained = getattr(self.hparams, 'pretrained', True)

        ehr_params = {
            'input_size': self.hparams.input_dim,
            'num_classes': self.num_classes,
            'hidden_size': self.hparams.dim,
            'dropout': self.hparams.dropout,
        }

        if ehr_encoder_type == 'lstm':
            ehr_params.update({
                'num_layers': getattr(self.hparams, 'ehr_num_layers', 2),
                'bidirectional': getattr(self.hparams, 'ehr_bidirectional', True)
            })
        elif ehr_encoder_type == 'transformer':
            ehr_params.update({
                'd_model': ehr_params.pop('hidden_size'),
                'n_head': getattr(self.hparams, 'ehr_n_head', 8),
                'n_layers': getattr(self.hparams, 'ehr_n_layers', 2),
            })

        self.ehr_model = create_ehr_encoder(encoder_type=ehr_encoder_type, **ehr_params)
        self.ehr_hidden_dim = self.hparams.dim

        if cxr_encoder_type == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            self.cxr_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.cxr_backbone.fc = nn.Identity()
            self.cxr_feat_dim = 2048
            self.cxr_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported CXR encoder type: {cxr_encoder_type}. Supported types: 'resnet50'")

        self.cxr_classifier = nn.Linear(self.cxr_feat_dim, self.num_classes)
        self.ehr_classifier = nn.Linear(self.ehr_hidden_dim, self.num_classes)
        self.mmtm_layers = nn.ModuleList([
            MMTMLayer(self.cxr_channels[i], self.ehr_hidden_dim, self.hparams.mmtm_ratio)
            for i in range(5)
        ])
        feats_dim = 2 * self.cxr_feat_dim
        self.joint_cls = nn.Sequential(
            nn.Linear(feats_dim, self.num_classes),
        )
        self.projection = nn.Linear(self.ehr_hidden_dim, self.cxr_feat_dim)
        self.align_loss_fn = CosineLoss()
        self.kl_loss = KLDivLoss()

        print(f"MMTM model initialized:")
        print(f"  - EHR encoder: {ehr_encoder_type}, hidden_dim: {self.ehr_hidden_dim}")
        print(f"  - CXR encoder: {cxr_encoder_type}, feat_dim: {self.cxr_feat_dim}")
        print(f"  - Task: {self.task}, Pretrained: {pretrained}")
        print(f"  - MMTM applied at layer: {self.hparams.layer_after}")
        print(f"  - CXR channels: {self.cxr_channels}")

    def _extract_cxr_features_progressive(self, img):
        features = []
        if hasattr(self.cxr_backbone, 'conv1'):
            x = self.cxr_backbone.conv1(img)
            x = self.cxr_backbone.bn1(x)
            x = self.cxr_backbone.relu(x)
            x = self.cxr_backbone.maxpool(x)
            features.append(x)
            x = self.cxr_backbone.layer1(x)
            features.append(x)
            x = self.cxr_backbone.layer2(x)
            features.append(x)
            x = self.cxr_backbone.layer3(x)
            features.append(x)
            x = self.cxr_backbone.layer4(x)
            features.append(x)
            final_feat = self.cxr_backbone.avgpool(x)
            final_feat = torch.flatten(final_feat, 1)
        else:
            final_feat = self.cxr_backbone(img)
            features = [final_feat.unsqueeze(-1).unsqueeze(-1)] * 5
        return features, final_feat

    def configure_optimizers(self):
        self.optimizer_visual = torch.optim.AdamW(
            [{'params': self.cxr_backbone.parameters()},
             {'params': self.mmtm_layers.parameters()}
            ],
            lr=self.hparams.lr
        )
        self.optimizer_ehr = torch.optim.AdamW(
            [{'params': self.ehr_model.parameters()},
             {'params': self.mmtm_layers.parameters()}
            ],
            lr=self.hparams.lr
        )
        self.optimizer_joint = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr
        )
        self.optimizer_early = torch.optim.AdamW(
            self.joint_cls.parameters(),
            lr=self.hparams.lr
        )
        self.scheduler_visual = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_visual,
            factor=0.5,
            patience=10,
            mode='min',
            verbose=True
        )
        self.scheduler_ehr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_ehr,
            factor=0.5,
            patience=10,
            mode='min',
            verbose=True
        )
        return None

    def forward(self, batch):
        x = batch['ehr_ts']
        seq_lengths = batch['seq_len']
        img = batch['cxr_imgs']
        y = batch['labels'].squeeze()

        ehr_feat, ehr_pred = self.ehr_model(x, seq_lengths, output_prob=False)
        ehr_unpacked = ehr_feat.unsqueeze(1).expand(-1, x.size(1), -1)

        cxr_features, cxr_final_feat = self._extract_cxr_features_progressive(img)

        for layer_idx, (mmtm_layer, cxr_feat) in enumerate(zip(self.mmtm_layers, cxr_features)):
            if (self.hparams.layer_after == layer_idx or 
                self.hparams.layer_after == -1):
                cxr_feat, ehr_unpacked = mmtm_layer(cxr_feat, ehr_unpacked)
                cxr_features[layer_idx] = cxr_feat

        if len(cxr_features) > 0:
            if hasattr(self.cxr_backbone, 'conv1'):
                cxr_final_feat = self.cxr_backbone.avgpool(cxr_features[-1])
                cxr_final_feat = torch.flatten(cxr_final_feat, 1)
            else:
                cxr_final_feat = cxr_features[-1].squeeze(-1).squeeze(-1)

        cxr_preds = self.cxr_classifier(cxr_final_feat)
        ehr_fused_feat = torch.mean(ehr_unpacked, dim=1)
        ehr_preds = self.ehr_classifier(ehr_fused_feat)
        late_average = (cxr_preds + ehr_preds) / 2
        projected = self.projection(ehr_fused_feat)
        align_loss = self.kl_loss(cxr_final_feat, projected)
        joint_feats = torch.cat([projected, cxr_final_feat], dim=1)
        joint_preds = self.joint_cls(joint_feats)

        output = {
            'predictions': torch.sigmoid(joint_preds),
            'logits': joint_preds,
            'labels': y,
            'loss': self.classification_loss(joint_preds, y),
            'mmtm_fusion': torch.sigmoid(joint_preds),
            'mmtm_fusion_scores': joint_preds,
            'cxr_predictions': cxr_preds,
            'ehr_predictions': ehr_preds,
            'late_fusion': late_average,
            'align_loss': align_loss,
        }
        return output

    def _step_with_optimizer(self, optimizer, batch, key):
        optimizer.zero_grad()
        out = self(batch)
        loss = out['loss']
        if key == 'align':
            loss = loss + out['align_loss']
        self.manual_backward(loss)
        optimizer.step()
        return out

    def training_step(self, batch, batch_idx):
        out_visual = self._step_with_optimizer(self.optimizer_visual, batch, 'visual')
        out_ehr = self._step_with_optimizer(self.optimizer_ehr, batch, 'ehr')
        out_joint = self._step_with_optimizer(self.optimizer_joint, batch, 'joint')
        out_early = self._step_with_optimizer(self.optimizer_early, batch, 'early')

        self.log_dict({
            'train/loss': out_joint['loss'].detach(),
            'train/align_loss': out_joint['align_loss'].detach()
        }, on_epoch=True, on_step=True, batch_size=out_joint['labels'].shape[0], sync_dist=True)

        return {
            "loss": out_joint['loss'], 
            "pred": out_joint['predictions'].detach(), 
            "labels": out_joint['labels'].detach()
        }

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            out = self.forward(batch)
        return self._val_test_shared_step(batch, self.test_info)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            out = self.forward(batch)
        out = self._val_test_shared_step(batch, self.val_info)
        self.log_dict({
            'val/loss': out['loss'].detach(),
            'loss/validation': out['loss'].detach(),
        }, on_epoch=True, on_step=False, batch_size=out['labels'].shape[0], sync_dist=True)
        return out['loss'].detach()

    def on_validation_epoch_end(self):
        if hasattr(self, 'val_info') and len(self.val_info['predictions']) > 0:
            scores = self._val_test_epoch_end(self.val_info, clear_cache=True)
            for metric_name, metric_value in scores.items():
                if not isinstance(metric_value, list):
                    self.log(metric_name, metric_value, 
                            on_epoch=True, on_step=False, 
                            prog_bar=True, logger=True, sync_dist=True)
            current_prauc = scores.get('overall/PRAUC', 0.0)
            if not hasattr(self, 'best_prauc'):
                self.best_prauc = -1.0
                self.best_model_path = None
            if current_prauc > self.best_prauc:
                self.best_prauc = current_prauc
                if self.trainer.global_rank == 0:
                    if hasattr(self.logger, 'log_dir'):
                        log_dir = self.logger.log_dir
                    elif hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log_dir'):
                        log_dir = self.logger.experiment.log_dir
                    else:
                        log_dir = './checkpoints'
                    import os
                    os.makedirs(log_dir, exist_ok=True)
                    self.best_model_path = os.path.join(log_dir, f'best_model_epoch_{self.current_epoch:02d}_prauc_{current_prauc:.4f}.ckpt')
                    self.trainer.save_checkpoint(self.best_model_path)
                    print(f"🎯 Manually saved best model: {self.best_model_path} (PRAUC: {current_prauc:.4f})")
            print(f"✅ Validation metrics: overall/PRAUC = {current_prauc:.4f} (Best: {self.best_prauc:.4f})")
        if hasattr(self, 'scheduler_visual'):
            val_loss = self.trainer.callback_metrics.get('loss/validation_epoch', 0)
            self.scheduler_visual.step(val_loss)
        if hasattr(self, 'scheduler_ehr'):
            val_loss = self.trainer.callback_metrics.get('loss/validation_epoch', 0)
            self.scheduler_ehr.step(val_loss)

    def load_state(self):
        if hasattr(self.hparams, 'load_state') and self.hparams.load_state:
            try:
                state_dict = torch.load(self.hparams.load_state, map_location='cpu')
                self.load_state_dict(state_dict['state_dict'])
                print(f"Successfully loaded model state from {self.hparams.load_state}")
            except Exception as e:
                print(f"Failed to load model state: {e}")

    def _shared_step(self, batch):
        return self.forward(batch)

    def _val_test_shared_step(self, batch, cache):
        with torch.no_grad():
            out = self.forward(batch)

        cache['predictions'].append(out['predictions'].detach().cpu())
        cache['labels'].append(out['labels'].detach().cpu())

        if 'meta_attrs' in batch:
            if 'meta_attrs' not in cache:
                cache['meta_attrs'] = []
            meta_attrs_list = batch['meta_attrs'].to_dict('records')
            cache['meta_attrs'].extend(meta_attrs_list)

        if 'groups' in batch:
            cache['groups'].extend(batch['groups'])

        return out
