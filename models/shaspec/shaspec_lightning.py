import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np

from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from ..base.base_encoder import create_ehr_encoder, create_cxr_encoder
from .shaspec_components import (
    CompositionalLayer, FusionClassifier, 
    MultiModalTransformerSharedEncoder
)

@ModelRegistry.register('shaspec')
class ShaSpec(BaseFuseTrainer):
    """
    ShaSpec - Shared-Specific Feature Modeling for Multimodal Learning with Missing Modality
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
            raise ValueError(f"Unsupported task: {self.task}. Only 'mortality', 'phenotype', and 'los' are supported.")

        self.feat_dim = getattr(self.hparams, 'dim', 256)
        self.input_dim = getattr(self.hparams, 'input_dim', 498)
        self.dropout = getattr(self.hparams, 'dropout', 0.2)
        self.weight_std = getattr(self.hparams, 'weight_std', True)

        if not hasattr(self.hparams, 'ehr_encoder'):
            self.hparams.ehr_encoder = 'lstm'
        if not hasattr(self.hparams, 'cxr_encoder'):
            self.hparams.cxr_encoder = 'resnet50'
        if not hasattr(self.hparams, 'pretrained'):
            self.hparams.pretrained = True

        self._init_model_components()

    def _init_model_components(self):
        ehr_encoder_type = getattr(self.hparams, 'ehr_encoder', 'lstm').lower()
        cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50').lower()
        pretrained = getattr(self.hparams, 'pretrained', True)

        self.shared_enc = MultiModalTransformerSharedEncoder(
            feat_dim=self.feat_dim,
            nhead=getattr(self.hparams, 'nhead', 8),
            num_layers=getattr(self.hparams, 'num_layers', 3),
            dropout=self.dropout,
            ehr_input_dim=self.input_dim,
            max_seq_len=getattr(self.hparams, 'max_seq_len', 500)
        )

        ehr_params = {
            'input_size': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_size': self.feat_dim,
            'dropout': self.dropout,
        }
        if ehr_encoder_type == 'lstm':
            ehr_params.update({
                'num_layers': getattr(self.hparams, 'ehr_num_layers', getattr(self.hparams, 'layers', 2)),
                'bidirectional': getattr(self.hparams, 'ehr_bidirectional', True)
            })
        elif ehr_encoder_type == 'transformer':
            ehr_params.update({
                'd_model': ehr_params.pop('hidden_size'),
                'n_head': getattr(self.hparams, 'ehr_n_head', 8),
                'n_layers': getattr(self.hparams, 'ehr_n_layers', 2),
            })

        self.ehr_enc = create_ehr_encoder(encoder_type=ehr_encoder_type, **ehr_params)

        cxr_params = {
            'hidden_size': self.feat_dim,
            'pretrained': pretrained
        }
        self.cxr_enc = create_cxr_encoder(encoder_type=cxr_encoder_type, **cxr_params)

        self.compos_layer = CompositionalLayer(
            feat_dim=self.feat_dim,
            weight_std=self.weight_std
        )
        self.fusion_classifier = FusionClassifier(
            feat_dim=self.feat_dim,
            num_classes=self.num_classes,
            dropout=self.dropout
        )
        self.dom_classifier = nn.Linear(
            in_features=self.feat_dim,
            out_features=2,
            bias=True
        )

        self.alpha = getattr(self.hparams, 'alpha', 0.1)
        self.beta = getattr(self.hparams, 'beta', 0.02)

        print(f"ShaSpec model initialized for {self.task} task with {self.num_classes} classes")
        print(f"  - EHR encoder: {ehr_encoder_type}, hidden_dim: {self.feat_dim}")
        print(f"  - CXR encoder: {cxr_encoder_type}, feat_dim: {self.feat_dim}")
        print(f"  - Task: {self.task}, Pretrained: {pretrained}")
        print(f"ShaSpec initialized with alpha={self.alpha}, beta={self.beta}")
        print("Using MultiModalTransformerSharedEncoder for direct raw input processing")

    def forward(self, batch):
        ehr_data = batch['ehr_ts']
        seq_lengths = batch['seq_len']
        cxr_img = batch['cxr_imgs']
        valid_cxr = batch.get('has_cxr', torch.ones(cxr_img.size(0), dtype=torch.bool, device=cxr_img.device))
        y = batch['labels']

        N = ehr_data.size(0)
        C = 2

        shared_ft, shared_gft = self.shared_enc(ehr_data, cxr_img, seq_lengths, valid_cxr)

        ehr_shared_ft = shared_ft[0::2]
        cxr_shared_ft = shared_ft[1::2]

        ehr_ft, _ = self.ehr_enc(ehr_data, seq_lengths, output_prob=False)
        cxr_ft = self.cxr_enc(cxr_img)

        general_shared_ft = ehr_shared_ft

        ehr_fused_ft = self.compos_layer(ehr_shared_ft, ehr_ft)

        cxr_fused_ft = torch.zeros_like(ehr_fused_ft)
        for i in range(N):
            if valid_cxr[i]:
                cxr_fused_ft[i] = self.compos_layer(
                    cxr_shared_ft[i:i+1], 
                    cxr_ft[i:i+1]
                ).squeeze(0)
            else:
                cxr_fused_ft[i] = general_shared_ft[i]

        fused_ft = shared_ft.clone()
        fused_ft[0::2] = ehr_fused_ft
        fused_ft[1::2] = cxr_fused_ft

        combined_feat = torch.mean(fused_ft.view(N, C, self.feat_dim), dim=1)

        logits = self.fusion_classifier(combined_feat)
        predictions = torch.sigmoid(logits)

        spec_gft = shared_gft.clone()
        spec_gft[0::2] = shared_gft[0::2]
        spec_gft[1::2] = shared_gft[1::2]

        output = {
            'predictions': predictions,
            'labels': y,
            'shared_ft': shared_ft,
            'spec_gft': spec_gft,
            'valid_cxr': valid_cxr,
            'mode_split': [0, 1] if torch.all(valid_cxr) else [0]
        }

        output['loss'] = self.compute_total_loss(output, logits, y)
        return output

    def compute_total_loss(self, output, logits, labels):
        classification_loss = self.classification_loss(logits, labels)

        term_shared = self._compute_shared_consistency_loss_original_style(
            output['shared_ft'], 
            output['valid_cxr']
        )

        term_spec = self._compute_domain_classification_loss_original_style(
            output['spec_gft']
        )

        total_loss = classification_loss + self.alpha * term_shared + self.beta * term_spec

        return total_loss

    def _compute_shared_consistency_loss_original_style(self, shared_ft, valid_cxr):
        N = len(valid_cxr)
        if valid_cxr.dtype != torch.bool:
            valid_cxr = valid_cxr.bool()

        ehr_shared = shared_ft[0::2]
        cxr_shared = shared_ft[1::2]

        valid_indices = torch.where(valid_cxr)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=shared_ft.device, requires_grad=True)

        ehr_shared_valid = ehr_shared[valid_indices]
        cxr_shared_valid = cxr_shared[valid_indices]

        distribution_loss = nn.L1Loss()
        term_shared = distribution_loss(ehr_shared_valid, cxr_shared_valid)

        return term_shared

    def _compute_domain_classification_loss_original_style(self, spec_gft):
        N = spec_gft.size(0) // 2
        device = spec_gft.device

        spec_labels = torch.zeros(N * 2, dtype=torch.long, device=device)
        spec_labels[0::2] = 0
        spec_labels[1::2] = 1

        spec_logits = self.dom_classifier(spec_gft.squeeze())

        loss_domain_cls = nn.CrossEntropyLoss()
        term_spec = loss_domain_cls(spec_logits, spec_labels)

        return term_spec

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