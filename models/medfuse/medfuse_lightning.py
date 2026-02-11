import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from .medfuse_components import Fusion, LSTM, CXRModels
from .ehr_encoder import DisentangledEHRTransformer
from torchvision.models import resnet50, ResNet50_Weights


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
        if self.hparams.drfuse_encoder:
            self.ehr_model = DisentangledEHRTransformer(input_size=24, num_classes=self.num_classes,
                                            d_model=self.hparams.dim, n_head=4,
                                            n_layers_feat=1, n_layers_shared=1,
                                            n_layers_distinct=1,
                                            dropout=self.hparams.dropout,simple=True)
            self.cxr_model= resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.cxr_model.fc = nn.Linear(in_features=2048, out_features=self.hparams.dim)
        else: 
            self.ehr_model = LSTM(
                input_dim=self.hparams.input_dim,
                num_classes=self.num_classes,
                hidden_dim=self.hparams.dim,
                dropout=self.hparams.dropout,
                layers=self.hparams.layers
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