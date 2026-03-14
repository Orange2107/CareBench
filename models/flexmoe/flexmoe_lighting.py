import torch
import torch.nn.functional as F
import numpy as np
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base.base_encoder import create_ehr_encoder, create_cxr_encoder
from .flexmoe_components import FlexMoE
from .moe_module import *
from itertools import combinations

class FlexMoEPatchAdapter(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_patches, dropout=0.3):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.projection = torch.nn.Linear(input_dim, hidden_dim * num_patches)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.projection(x)
        x = x.view(batch_size, self.num_patches, self.hidden_dim)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

class OptimizedFlexMoEModel(torch.nn.Module):
    def __init__(
        self,
        num_modalities,
        full_modality_index,
        input_dims,
        hidden_dim,
        output_dim,
        num_layers,
        num_layers_pred,
        num_experts,
        num_routers,
        top_k,
        num_heads=2,
        dropout=0.5,
        num_patches=16,
        ehr_encoder='lstm',
        cxr_encoder='resnet50',
        pretrained=True,
        task='phenotype',
        **encoder_kwargs
    ):
        super(OptimizedFlexMoEModel, self).__init__()
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        ehr_params = {
            'input_size': input_dims[0],
            'num_classes': output_dim,
            'hidden_size': hidden_dim,
            'dropout': dropout,
        }
        if ehr_encoder == 'lstm':
            ehr_params.update({
                'num_layers': encoder_kwargs.get('ehr_num_layers', 2),
                'bidirectional': encoder_kwargs.get('ehr_bidirectional', False)
            })
        elif ehr_encoder == 'transformer':
            ehr_params.update({
                'd_model': ehr_params.pop('hidden_size'),
                'n_head': encoder_kwargs.get('ehr_n_head', 8),
                'n_layers': encoder_kwargs.get('ehr_n_layers', 2),
            })
        self.ehr_encoder = create_ehr_encoder(encoder_type=ehr_encoder, **ehr_params)
        self.ehr_patch_adapter = FlexMoEPatchAdapter(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            dropout=dropout
        )
        if len(input_dims) > 1:
            cxr_params = {
                'hidden_size': hidden_dim,
                'pretrained': pretrained,
                'hf_model_id': encoder_kwargs.get('hf_model_id', 'codewithdark/vit-chest-xray'),
                'freeze_vit': encoder_kwargs.get('freeze_vit', True),
                'bias_tune': encoder_kwargs.get('bias_tune', False),
                'partial_layers': encoder_kwargs.get('partial_layers', 0),
            }
            self.cxr_encoder = create_cxr_encoder(encoder_type=cxr_encoder, **cxr_params)
            self.cxr_patch_adapter = FlexMoEPatchAdapter(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_patches=num_patches,
                dropout=dropout
            )
        self.flexmoe = FlexMoE(
            num_modalities=num_modalities,
            full_modality_index=full_modality_index,
            num_patches=num_patches,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_layers_pred=num_layers_pred,
            num_experts=num_experts,
            num_routers=num_routers,
            top_k=top_k,
            num_heads=num_heads,
            dropout=dropout
        )
        self.missing_embeds = torch.nn.Parameter(
            torch.randn(
                (2**self.num_modalities)-1,
                self.num_modalities,
                self.num_patches,
                self.hidden_dim,
                dtype=torch.float, 
            ), 
            requires_grad=True
        )
        
    def forward(self, inputs, expert_indices=None, has_modality=None):
        processed_inputs = []
        batch_size = inputs[0].size(0) if inputs[0] is not None else inputs[1].size(0)
        if inputs[0] is not None:
            ehr_data = inputs[0]
            seq_lengths = torch.sum((ehr_data.abs().sum(-1) > 0).long(), dim=1)
            seq_lengths = torch.clamp(seq_lengths, min=1)
            ehr_feat, _ = self.ehr_encoder(ehr_data, seq_lengths, output_prob=False)
            ehr_patches = self.ehr_patch_adapter(ehr_feat)
            if has_modality is not None and has_modality[0] is not None:
                mask = has_modality[0].bool()
                final_ehr = torch.zeros_like(ehr_patches)
                if mask.any():
                    final_ehr[mask] = ehr_patches[mask]
                if (~mask).any():
                    final_ehr[~mask] = self.missing_embeds[expert_indices[~mask].long(), 0]
                processed_inputs.append(final_ehr)
            else:
                processed_inputs.append(ehr_patches)
        else:
            ehr_patches = self.missing_embeds[expert_indices.long(), 0]
            processed_inputs.append(ehr_patches)
        if len(inputs) > 1 and inputs[1] is not None:
            cxr_feat = self.cxr_encoder(inputs[1])
            cxr_patches = self.cxr_patch_adapter(cxr_feat)
            if has_modality is not None and has_modality[1] is not None:
                mask = has_modality[1].bool()
                final_cxr = torch.zeros_like(cxr_patches)
                if mask.any():
                    final_cxr[mask] = cxr_patches[mask]
                if (~mask).any():
                    final_cxr[~mask] = self.missing_embeds[expert_indices[~mask].long(), 1]
                processed_inputs.append(final_cxr)
            else:
                processed_inputs.append(cxr_patches)
        else:
            cxr_patches = self.missing_embeds[expert_indices.long(), 1]
            processed_inputs.append(cxr_patches)
        return self.flexmoe(*processed_inputs, expert_indices=expert_indices)
    
    def gate_loss(self):
        return self.flexmoe.gate_loss()
    
    def assign_expert(self, combination):
        return self.flexmoe.assign_expert(combination)
    
    def set_full_modality(self, is_full_modality):
        self.flexmoe.set_full_modality(is_full_modality)

@ModelRegistry.register('flexmoe')
class FlexMoELightning(BaseFuseTrainer):
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
        if not hasattr(self.hparams, 'ehr_encoder'):
            self.hparams.ehr_encoder = 'lstm'
        if not hasattr(self.hparams, 'cxr_encoder'):
            self.hparams.cxr_encoder = 'resnet50'
        if not hasattr(self.hparams, 'pretrained'):
            self.hparams.pretrained = True
        self.num_modalities = 2
        self.combination_to_index = self._create_combination_index(self.num_modalities)
        self._init_model_components()
        
    def _create_combination_index(self, num_modalities):
        combinations_list = []
        for r in range(1, num_modalities + 1):
            combinations_list.extend(combinations(range(num_modalities), r))
        return {tuple(sorted(comb)): idx for idx, comb in enumerate(combinations_list)}
        
    def _init_model_components(self):
        ehr_encoder_type = getattr(self.hparams, 'ehr_encoder', 'lstm').lower()
        cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50').lower()
        pretrained = getattr(self.hparams, 'pretrained', True)
        ehr_dim = self.hparams.input_dim
        cxr_dim = getattr(self.hparams, 'cxr_dim', 3 * 224 * 224)
        input_dims = [ehr_dim, cxr_dim]
        full_modality_combo = tuple(range(self.num_modalities))
        full_modality_index = self.combination_to_index[full_modality_combo]
        encoder_kwargs = {
            'ehr_num_layers': getattr(self.hparams, 'ehr_num_layers', 2),
            'ehr_bidirectional': getattr(self.hparams, 'ehr_bidirectional', False),
            'ehr_n_head': getattr(self.hparams, 'ehr_n_head', 8),
            'ehr_n_layers': getattr(self.hparams, 'ehr_n_layers', 2),
        }
        self.model = OptimizedFlexMoEModel(
            num_modalities=self.num_modalities,
            full_modality_index=full_modality_index,
            input_dims=input_dims,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=self.num_classes,
            num_layers=self.hparams.num_layers,
            num_layers_pred=self.hparams.num_layers_pred,
            num_experts=self.hparams.num_experts,
            num_routers=self.hparams.num_routers,
            top_k=self.hparams.top_k,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
            num_patches=self.hparams.num_patches,
            ehr_encoder=ehr_encoder_type,
            cxr_encoder=cxr_encoder_type,
            pretrained=pretrained,
            task=self.task,
            **encoder_kwargs
        )
        print(f"Flex-MoE model initialized for {self.task} task")
        print(f"  - EHR encoder: {ehr_encoder_type}")
        print(f"  - CXR encoder: {cxr_encoder_type}")
        print(f"  - Pretrained: {pretrained}")

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
        ehr_data = batch['ehr_ts']
        ehr_masks = batch['ehr_masks']
        cxr_data = batch['cxr_imgs']
        batch_size = ehr_data.shape[0]
        has_ehr = ehr_masks.any(dim=(1, 2))
        has_cxr = batch.get('has_cxr', torch.ones(batch_size, dtype=torch.bool, device=self.device))
        expert_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            available_modalities = []
            if has_ehr[i]:
                available_modalities.append(0)
            if has_cxr[i]:
                available_modalities.append(1)
            combo_tuple = tuple(sorted(available_modalities))
            expert_indices[i] = self.combination_to_index[combo_tuple]
        is_full_modality = has_ehr.all() and has_cxr.all()
        self.model.set_full_modality(is_full_modality)
        inputs = [ehr_data]
        if has_cxr.any():
            inputs.append(cxr_data)
        else:
            inputs.append(None)
        logits = self.model(
            inputs=inputs,
            expert_indices=expert_indices,
            has_modality=[has_ehr, has_cxr]
        )
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        labels = batch['labels'].squeeze(-1)
        task_loss = self.classification_loss(logits, labels)
        gate_loss = self.model.gate_loss()
        if self.training and hasattr(self, 'current_epoch'):
            progress = min(self.current_epoch / self.hparams.epochs, 1.0)
            gate_loss_weight = self.hparams.gate_loss_weight * (1.0 - 0.5 * progress)
        else:
            gate_loss_weight = self.hparams.gate_loss_weight
        total_loss = task_loss + gate_loss_weight * gate_loss
        return {
            'loss': total_loss,
            'ce_loss': task_loss,
            'gate_loss': gate_loss,
            'gate_loss_weight': gate_loss_weight,
            'predictions': logits,
            'labels': labels,
            'expert_indices': expert_indices,
            'has_ehr': has_ehr,
            'has_cxr': has_cxr
        }
        
    def training_step(self, batch, batch_idx):
        out = self(batch)
        expert_counts = torch.bincount(
            out['expert_indices'],
            minlength=3
        )
        expert_dist = expert_counts.float() / expert_counts.sum() if expert_counts.sum() > 0 else expert_counts.float()
        log_dict = {
            'train/loss': out['loss'].detach(),
            'train/ce_loss': out['ce_loss'].detach(),
            'train/gate_loss': out['gate_loss'].detach(),
            'train/gate_loss_weight': out['gate_loss_weight'],
        }
        for i, freq in enumerate(expert_dist):
            log_dict[f'train/expert_{i}_usage'] = freq.item()
        log_dict['train/ehr_available'] = out['has_ehr'].float().mean().item()  
        log_dict['train/cxr_available'] = out['has_cxr'].float().mean().item()
        self.log_dict(log_dict, on_epoch=True, on_step=True, 
                     batch_size=batch['labels'].shape[0], sync_dist=True)
        return {"loss": out['loss'], "pred": out['predictions'].detach(), "labels": out['labels'].detach()}

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        out = self(batch)
        self.log_dict({
            'validation_epoch/loss': out['loss'].detach(),
            'validation_epoch/ce_loss': out['ce_loss'].detach(),
            'validation_epoch/gate_loss': out['gate_loss'].detach(),
        }, on_epoch=True, batch_size=batch['labels'].shape[0], sync_dist=True)
        return out['loss'].detach()
