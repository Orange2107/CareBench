import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from .healnet_adapters import build_cxr_adapter, build_ehr_adapter
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
        self.ehr_adapter = build_ehr_adapter(self.hparams)
        self.cxr_adapter = build_cxr_adapter(self.hparams)
        channel_dims = [
            self.ehr_adapter.get_channel_dim(),
            self.cxr_adapter.get_channel_dim(),
        ]
        num_spatial_axes = [
            self.ehr_adapter.get_num_spatial_axes(),
            self.cxr_adapter.get_num_spatial_axes(),
        ]

        self.model = HealNet(
            n_modalities=2,
            channel_dims=channel_dims,
            num_spatial_axes=num_spatial_axes,
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
        
        print(
            "HealNet model initialized in End-to-End training mode, "
            f"focused on {self.task} task "
            f"(ehr_input_mode={getattr(self.hparams, 'ehr_input_mode', 'raw')}, "
            f"cxr_input_mode={getattr(self.hparams, 'cxr_input_mode', 'raw')})"
        )

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
    
    def _normalize_has_cxr(self, has_cxr, batch_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(has_cxr, list):
            has_cxr = torch.tensor(has_cxr, device=device)
        elif not isinstance(has_cxr, torch.Tensor):
            has_cxr = torch.tensor(has_cxr, device=device)
        else:
            has_cxr = has_cxr.to(device)
        has_cxr = has_cxr.view(batch_size, -1)
        return has_cxr.any(dim=1).bool()

    def _run_healnet_subset(self, ehr_ts, seq_lengths, cxr_imgs=None):
        ehr_tensor = self.ehr_adapter(ehr_ts, seq_lengths)
        cxr_tensor = None if cxr_imgs is None else self.cxr_adapter(cxr_imgs)
        return self.model([ehr_tensor, cxr_tensor])

    def _predict_with_missing_modalities(self, x, seq_lengths, img, has_cxr):
        batch_size = x.shape[0]
        pred = x.new_zeros((batch_size, self.num_classes), dtype=torch.float32)

        if has_cxr.any():
            paired_mask = has_cxr
            pred[paired_mask] = self._run_healnet_subset(
                x[paired_mask],
                seq_lengths[paired_mask],
                img[paired_mask],
            )

        if (~has_cxr).any():
            ehr_only_mask = ~has_cxr
            pred[ehr_only_mask] = self._run_healnet_subset(
                x[ehr_only_mask],
                seq_lengths[ehr_only_mask],
                None,
            )

        return pred

    def forward(self, batch):
        x = batch['ehr_ts']
        seq_lengths = batch['seq_len']
        if isinstance(seq_lengths, list):
            seq_lengths = torch.tensor(seq_lengths, device=x.device)
        elif not isinstance(seq_lengths, torch.Tensor):
            seq_lengths = torch.tensor(seq_lengths, device=x.device)
        else:
            seq_lengths = seq_lengths.to(x.device)
        img = batch['cxr_imgs']
        has_cxr = self._normalize_has_cxr(batch['has_cxr'], x.shape[0], x.device)
        y = batch['labels'].squeeze(-1)

        pred = self._predict_with_missing_modalities(x, seq_lengths, img, has_cxr)

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
