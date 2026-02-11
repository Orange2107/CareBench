import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from .healnet_components import HealNet

@ModelRegistry.register('healnet')
class HealNetLightning(BaseFuseTrainer):
    """
    HealNet Lightning Module - Integration with benchmark framework.
    This module adapts the HealNet model to the benchmark's LightningModule framework.
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
        self.model = HealNet(
            n_modalities=2,
            channel_dims=[self.hparams.input_dim, 3],
            num_spatial_axes=[1, 2],
            out_dims=self.num_classes,
            depth=self.hparams.depth,
            num_freq_bands=self.hparams.num_freq_bands,
            max_freq=self.hparams.max_freq,
            l_c=self.hparams.latent_channels,
            l_d=self.hparams.latent_dim,
            x_heads=self.hparams.cross_heads,
            l_heads=self.hparams.latent_heads,
            cross_dim_head=self.hparams.cross_dim_head,
            latent_dim_head=self.hparams.latent_dim_head,
            attn_dropout=self.hparams.attn_dropout,
            ff_dropout=self.hparams.ff_dropout,
            weight_tie_layers=self.hparams.weight_tie_layers,
            fourier_encode_data=self.hparams.fourier_encode_data,
            self_per_cross_attn=self.hparams.self_per_cross_attn,
            final_classifier_head=self.hparams.final_classifier_head,
            snn=self.hparams.snn
        )
        
        print(f"HealNet model initialized in End-to-End training mode, focused on {self.task} task")

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
    
    def forward(self, batch):
        x = batch['ehr_ts']
        seq_lengths = batch['seq_len']
        img = batch['cxr_imgs']
        has_cxr = batch['has_cxr']
        y = batch['labels'].squeeze(-1)
        
        ehr_tensor = x
        cxr_tensor = img.permute(0, 2, 3, 1)
        tensors = [ehr_tensor, cxr_tensor]
        
        pred = self.model(tensors)
        
        if self.task == 'mortality' and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
            
        loss = self.classification_loss(pred, y)
        
        output = {
            'loss': loss,
            'predictions': pred,
            'labels': y
        }
        
        return output
        
    def training_step(self, batch, batch_idx):
        out = self(batch)
        
        self.log_dict({'train/loss': out['loss'].detach()}, 
                     on_epoch=True, on_step=True, 
                     batch_size=out['labels'].shape[0],
                     sync_dist=True)
                     
        return {"loss": out['loss'], "pred": out['predictions'].detach(), "labels": out['labels'].detach()}