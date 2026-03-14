import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
from .umse_components import *
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from torch.optim.lr_scheduler import ReduceLROnPlateau

import copy
import yaml

import warnings
warnings.filterwarnings("ignore")


@ModelRegistry.register('umse')
class UMSE(BaseFuseTrainer):
    def __init__(self,hparams):
        
        super().__init__()
        self.save_hyperparameters(hparams)
        self.task = self.hparams.task
        self.max_ehr_len=self.hparams.max_ehr_len

        # Set task-specific number of classes
        if self.task == 'phenotype':
            self.output_dim = self.hparams.num_classes
        elif self.task == 'mortality':
            self.output_dim = 1  # Binary classification
        elif self.task == 'los':
            self.output_dim = 7  # LoS has 7 classes (bins 2-8, excluding 0,1)
        else:
            raise ValueError(f"Unsupported task: {self.task}. Only 'mortality', 'phenotype', and 'los' are supported")
        
        self._init_model_components()
        

    def _init_model_components(self):
        
       self.model=MLHC(
            d_model=self.hparams.d_model,
            output_dim=self.output_dim,
            variables_num=self.hparams.variables_num,
            num_layers=self.hparams.num_layers,
            batch_size=self.hparams.batch_size,
            num_heads=self.hparams.num_heads,
            n_modality=self.hparams.n_modality,
            bottlenecks_n=self.hparams.bottlenecks_n,
            dropout_rate=self.hparams.dropout,
            max_ehr_len=self.max_ehr_len,
            cxr_encoder=getattr(self.hparams, 'cxr_encoder', 'patch_embed'),
            pretrained=getattr(self.hparams, 'pretrained', True),
            hf_model_id=getattr(self.hparams, 'hf_model_id', "codewithdark/vit-chest-xray"),
            freeze_vit=getattr(self.hparams, 'freeze_vit', True),
            bias_tune=getattr(self.hparams, 'bias_tune', False),
            partial_layers=getattr(self.hparams, 'partial_layers', 0),
       )
       
    
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

    # reg_ts,  last_cxr,  last_cxr_time
    def forward(self, batch):
        """Forward pass adapting HealNet to our multimodal clinical data"""
        # Extract inputs
        ehr_ts = batch['ehr_ts']  # [batch_size, seq_len, features]
        # cut ehr_ts to max_ehr_len
        ehr_ts = ehr_ts[:,:self.max_ehr_len]
        ehr_mask = (~(ehr_ts == 0).all(dim=-1))  # [batch_size, seq_len]
        
        last_cxr = batch['cxr_imgs']  # [batch_size, channels, height, width]
        last_cxr_time=batch['cxr_times']
        has_cxr = batch['has_cxr'].int()
        y = batch['labels'].squeeze(-1)
        
        
       
        pred = self.model(ehr_ts,ehr_mask,has_cxr,last_cxr,last_cxr_time)
        
        # For mortality task, ensure output shape matches labels
        if self.task == 'mortality' and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)  # [B,1] -> [B]
            
        # Calculate loss
        loss = self.classification_loss(pred, y)
        
        # Return output dict compatible with BaseFuseTrainer
        output = {
            'loss': loss,
            'predictions': pred,
            'labels': y
        }
        
        return output

    def training_step(self, batch, batch_idx):
        """Training step"""
        out = self(batch)
        
        # Log loss
        self.log_dict({'train/loss': out['loss'].detach()}, 
                     on_epoch=True, on_step=True, 
                     batch_size=out['labels'].shape[0],
                     sync_dist=True)
                     
        return {"loss": out['loss'], "pred": out['predictions'].detach(), "labels": out['labels'].detach()}
        
    
