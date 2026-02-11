import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
from models.registry import ModelRegistry
 
from ..base.base_encoder import TransformerEncoder, LSTMEncoder
from ..base import BaseFuseTrainer

@ModelRegistry.register('latefusion')
class LateFusion(BaseFuseTrainer):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        if self.hparams.task == 'phenotype':
            self.num_classes = self.hparams.num_classes
        elif self.hparams.task == 'mortality':
            self.num_classes = 1
        elif self.hparams.task == 'los':
            self.num_classes = 7  # LoS has 7 classes (bins 2-8, excluding 0,1)
        else:
            raise ValueError(f"Unsupported task: {self.hparams.task}. Only 'mortality', 'phenotype', and 'los' are supported")

        # EHR Encoder Selection
        ehr_encoder_type = getattr(self.hparams, 'ehr_encoder', 'transformer')
        if ehr_encoder_type.lower() == 'lstm':
            self.ehr_model = LSTMEncoder(
                input_size=self.hparams.input_dim, 
                num_classes=self.num_classes,
                hidden_size=self.hparams.hidden_size,
                num_layers=getattr(self.hparams, 'ehr_n_layers', 2),
                dropout=self.hparams.ehr_dropout,
                bidirectional=getattr(self.hparams, 'ehr_lstm_bidirectional', True)
            )
        elif ehr_encoder_type.lower() == 'transformer':
            self.ehr_model = TransformerEncoder(
                input_size=self.hparams.input_dim,
                num_classes=self.num_classes,
                d_model=getattr(self.hparams, 'hidden_size', 256),
                n_head=getattr(self.hparams, 'ehr_n_head', 8),
                n_layers=getattr(self.hparams, 'ehr_n_layers', 2),
                dropout=getattr(self.hparams, 'ehr_dropout', 0.3),
                max_len=getattr(self.hparams, 'max_len', 500)
            )
        else:
            raise ValueError(f"Unsupported EHR encoder type: {ehr_encoder_type}. Supported types: 'lstm', 'transformer'")

        # CXR Encoder Selection (ResNet50)
        cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50')
        pretrained = getattr(self.hparams, 'pretrained', True)
        
        if cxr_encoder_type.lower() == 'resnet50':
            self.cxr_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.cxr_model.fc = nn.Linear(in_features=2048, out_features=self.hparams.hidden_size)
        else:
            raise ValueError(f"Unsupported CXR encoder type: {cxr_encoder_type}. Supported types: 'resnet50'")

        # Simple late fusion: concatenate the two modality features and use a linear layer for final prediction
        self.final_pred_fc = nn.Linear(self.hparams.hidden_size * 2, self.num_classes)

        # Use BCEWithLogitsLoss for binary/multi-label tasks, CrossEntropyLoss for LoS
        if self.hparams.task == 'los':
            self.pred_criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            self.pred_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, data_dict):
        x = data_dict['ehr_ts']  # [batch, seq_lengths, input_dim]
        img = data_dict['cxr_imgs']
        seq_lengths = data_dict['seq_len']
        has_cxr = data_dict.get('has_cxr', None)

        # Encode both modalities
        feat_ehr, _ = self.ehr_model(x, seq_lengths)
        feat_cxr = self.cxr_model(img)

        # Late fusion: concatenate both features
        feat_final = torch.cat([feat_ehr, feat_cxr], dim=-1)
        pred_final = self.final_pred_fc(feat_final)

        outputs = {
            'feat_ehr_distinct': feat_ehr,
            'feat_cxr_distinct': feat_cxr,
            'feat_final': feat_final,
            'predictions': pred_final,
        }

        # Compute the loss
        if self.hparams.task == 'los':
            # For LoS: use CrossEntropyLoss (labels must be long and squeezed)
            loss = self.pred_criterion(pred_final, data_dict['labels'].long().squeeze())
        else:
            # For binary or multi-label tasks: use BCEWithLogitsLoss
            loss = self.pred_criterion(pred_final, data_dict['labels'])
        outputs['loss'] = loss

        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr
        )

        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=self.hparams.patience,
                mode='min',
                verbose=True
            ),
            "monitor": "loss/validation_epoch",
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

