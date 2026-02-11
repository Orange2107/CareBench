import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from .daft_components import DAFTBlock
from ..base.base_encoder import create_ehr_encoder, create_cxr_encoder
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import os

@ModelRegistry.register('daft')
class DAFT(BaseFuseTrainer):
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
            raise ValueError(f"Unsupported task: {self.task}. Only 'mortality', 'phenotype', and 'los' are supported.")

        if not hasattr(self.hparams, 'layer_after'):
            self.hparams.layer_after = 4
        if not hasattr(self.hparams, 'daft_activation'):
            self.hparams.daft_activation = 'linear'
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

        if ehr_encoder_type == 'lstm':
            self.ehr_model = create_ehr_encoder(
                encoder_type='lstm',
                input_size=self.hparams.input_dim,
                num_classes=self.num_classes,
                hidden_size=self.hparams.dim,
                dropout=self.hparams.dropout,
                num_layers=2,
                bidirectional=True
            )
            self.ehr_hidden_dim = self.hparams.dim
        elif ehr_encoder_type == 'transformer':
            self.ehr_model = create_ehr_encoder(
                encoder_type='transformer',
                input_size=self.hparams.input_dim,
                num_classes=self.num_classes,
                d_model=self.hparams.dim,
                n_head=getattr(self.hparams, 'ehr_n_head', 8),
                n_layers=getattr(self.hparams, 'ehr_n_layers', 2),
                dropout=self.hparams.dropout
            )
            self.ehr_hidden_dim = self.hparams.dim
        else:
            raise ValueError(f"Unsupported EHR encoder type: {ehr_encoder_type}. Supported types: 'lstm', 'transformer'")

        cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50').lower()
        pretrained = getattr(self.hparams, 'pretrained', True)

        if cxr_encoder_type == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            self.cxr_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.cxr_backbone.fc = nn.Identity()
            self.cxr_feat_dim = 2048
            self.cxr_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported CXR encoder type: {cxr_encoder_type}. Supported types: 'resnet50'")

        self.cxr_classifier = nn.Linear(self.cxr_feat_dim, self.num_classes)

        def calc_bottleneck_dim(cxr_dim, ehr_dim):
            total_input = cxr_dim + ehr_dim
            return max(16, min(128, total_input // 4))

        bottleneck_dims = [calc_bottleneck_dim(cxr_ch, self.ehr_hidden_dim) for cxr_ch in self.cxr_channels]

        self.daft_layers = nn.ModuleList([
            DAFTBlock(
                in_channels=self.ehr_hidden_dim,
                ndim_non_img=self.cxr_channels[i],
                bottleneck_dim=bottleneck_dims[i],
                location=0,
                activation=self.hparams.daft_activation
            ) for i in range(5)
        ])

        print(f"DAFT model initialized:")
        print(f"  - EHR encoder: {ehr_encoder_type}, hidden_dim: {self.ehr_hidden_dim}")
        print(f"  - CXR encoder: {cxr_encoder_type}, feat_dim: {self.cxr_feat_dim}")
        print(f"  - Task: {self.task}, Pretrained: {pretrained}")
        print(f"  - DAFT applied at layer: {self.hparams.layer_after}")
        print(f"  - CXR channels: {self.cxr_channels}")
        print(f"  - Bottleneck dims: {bottleneck_dims}")

        self.final_classifier = nn.Linear(self.ehr_hidden_dim, self.num_classes)

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

    def forward(self, batch):
        x = batch['ehr_ts']
        seq_lengths = batch['seq_len']
        img = batch['cxr_imgs']
        y = batch['labels'].squeeze()

        if self.hparams.ehr_encoder.lower() == 'lstm':
            ehr_feat, ehr_pred = self.ehr_model(x, seq_lengths, output_prob=False)
            ehr_unpacked = ehr_feat.unsqueeze(1).expand(-1, x.size(1), -1)
        elif self.hparams.ehr_encoder.lower() == 'transformer':
            ehr_feat, ehr_pred = self.ehr_model(x, seq_lengths, output_prob=False)
            ehr_unpacked = ehr_feat.unsqueeze(1).expand(-1, x.size(1), -1)

        cxr_features, cxr_final_feat = self._extract_cxr_features_progressive(img)

        for layer_idx, (daft_layer, cxr_feat) in enumerate(zip(self.daft_layers, cxr_features)):
            if (self.hparams.layer_after == layer_idx or
                self.hparams.layer_after == -1):
                ehr_unpacked = daft_layer(ehr_unpacked, cxr_feat)

        ehr_fused_feat = torch.mean(ehr_unpacked, dim=1)

        ehr_preds = self.final_classifier(ehr_fused_feat)
        cxr_preds = self.cxr_classifier(cxr_final_feat)

        ehr_probs = torch.sigmoid(ehr_preds)
        cxr_probs = torch.sigmoid(cxr_preds)

        output = {
            'daft_fusion': ehr_probs,
            'daft_fusion_scores': ehr_preds,
            'predictions': ehr_preds,
            'cxr_predictions': cxr_preds,
            'labels': y
        }

        output['loss'] = self.classification_loss(ehr_preds, y)

        return output

    def training_step(self, batch, batch_idx):
        out = self(batch)
        self.log_dict({'train/loss': out['loss'].detach()}, on_epoch=True, on_step=True,
                      batch_size=out['labels'].shape[0], sync_dist=True)
        return {"loss": out['loss'], "pred": out['predictions'].detach(), "labels": out['labels'].detach()}

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

    def load_state(self):
        if self.hparams.load_state:
            try:
                state_dict = torch.load(self.hparams.load_state, map_location='cpu')
                self.load_state_dict(state_dict['state_dict'])
                print(f"Successfully loaded model state from {self.hparams.load_state}")
            except Exception as e:
                print(f"Failed to load model state: {e}")
