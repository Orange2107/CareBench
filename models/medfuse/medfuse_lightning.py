import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from .medfuse_components import Fusion, LSTM, CXRModels
from .ehr_encoder import DisentangledEHRTransformer, TransformerEncoder
from ..base.base_encoder import create_cxr_encoder


class MedFuseTransformerAdapter(nn.Module):
    """Normalize TransformerEncoder outputs to match MedFuse's (pred, feat) convention."""

    def __init__(self, input_dim, num_classes, dim, n_head, n_layers, dropout, max_len):
        super().__init__()
        self.encoder = TransformerEncoder(
            input_size=input_dim,
            num_classes=num_classes,
            d_model=dim,
            n_head=n_head,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.feats_dim = dim

    def forward(self, x, seq_lengths):
        feat, prediction = self.encoder(x, seq_lengths, output_prob=True)
        return prediction, feat


@ModelRegistry.register('medfuse')
class MedFuse(BaseFuseTrainer):
    """
    MedFuse model LightningModule implementation - End-to-End training version:
    - Focused on clinical tasks (mortality and phenotype)
    - All components randomly initialized and trained together
    - Multiple fusion strategies (early, late, uni, lstm)
    - Automatically inherits evaluation and metric calculation logic from BaseFuseTrainer
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
        ehr_encoder_type = getattr(self.hparams, 'ehr_encoder', 'lstm').lower()
        self.hparams.drfuse_encoder = (ehr_encoder_type == 'drfuse')

        if self.hparams.drfuse_encoder:
            # DRFuse-style transformer encoder uses ehr_dropout
            # input_size should match the actual EHR feature dimension
            input_size = getattr(self.hparams, 'input_dim', 24)
            self.ehr_model = DisentangledEHRTransformer(
                input_size=input_size,
                num_classes=self.num_classes,
                d_model=self.hparams.dim, n_head=4,
                n_layers_feat=1, n_layers_shared=1,
                n_layers_distinct=1,
                dropout=getattr(self.hparams, 'ehr_dropout', 0.3),
                simple=True)
            cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50')
            pretrained = getattr(self.hparams, 'pretrained', True)
            self.cxr_model = create_cxr_encoder(
                encoder_type=cxr_encoder_type,
                hidden_size=self.hparams.dim,
                pretrained=pretrained,
                hf_model_id=getattr(self.hparams, 'hf_model_id', 'codewithdark/vit-chest-xray'),
                freeze_vit=getattr(self.hparams, 'freeze_vit', True),
                bias_tune=getattr(self.hparams, 'bias_tune', False),
                partial_layers=getattr(self.hparams, 'partial_layers', 0),
            )
        elif ehr_encoder_type == 'transformer':
            self.ehr_model = MedFuseTransformerAdapter(
                input_dim=self.hparams.input_dim,
                num_classes=self.num_classes,
                dim=self.hparams.dim,
                n_head=getattr(self.hparams, 'ehr_n_head', 8),
                n_layers=getattr(self.hparams, 'ehr_n_layers', 2),
                dropout=getattr(self.hparams, 'ehr_dropout', 0.3),
                max_len=getattr(self.hparams, 'max_len', 350),
            )
            self.cxr_model = CXRModels(self.hparams)
        else:
            # LSTM encoder uses lstm_dropout
            self.ehr_model = LSTM(
                input_dim=self.hparams.input_dim,
                num_classes=self.num_classes,
                hidden_dim=self.hparams.dim,
                dropout=getattr(self.hparams, 'lstm_dropout', 0.3),
                layers=getattr(self.hparams, 'lstm_layers', 1)
            )
            self.cxr_model = CXRModels(self.hparams)

        self.model = Fusion(self.hparams, self.ehr_model, self.cxr_model)
        
        print(f"MedFuse model initialized in End-to-End training mode, focused on {self.task} task")

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

    def forward(self, batch):
        x = batch['ehr_ts']
        seq_lengths = batch['seq_len']
        img = batch['cxr_imgs']
        pairs = batch['has_cxr']
        y = batch['labels'].squeeze()

        output = self.model(x, seq_lengths, img, pairs)
        pred = output[self.hparams.fusion_type].squeeze()

        loss = self.classification_loss(pred, y)

        if self.hparams.align > 0.0 and 'align_loss' in output:
            loss += self.hparams.align * output['align_loss']
            output['align_loss_logged'] = output['align_loss']

        output.update({
            'loss': loss,
            'predictions': pred,    
            'labels': y
        })

        return output

    def training_step(self, batch, batch_idx):
        out = self(batch)
        
        self.log_dict({'train/loss': out['loss'].detach()}, on_epoch=True, on_step=True, 
                      batch_size=out['labels'].shape[0], sync_dist=True)

        if 'align_loss_logged' in out:
            self.log_dict({'train/align_loss': out['align_loss_logged'].detach()}, on_epoch=True, 
                          on_step=True, batch_size=out['labels'].shape[0], sync_dist=True)

        return {"loss": out['loss'], "pred": out['predictions'].detach(), "labels": out['labels'].detach()}
