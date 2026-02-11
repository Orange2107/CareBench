import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from ..base.base_encoder import TransformerEncoder

@ModelRegistry.register('transformer')
class TransformerModel(BaseFuseTrainer):
    """
    Single-modal Transformer model for EHR data using base TransformerEncoder
    - Uses the base encoder's TransformerEncoder implementation
    - Supports both mortality and phenotype prediction tasks
    - Inherits training/validation/test steps from BaseFuseTrainer
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.task = self.hparams.task
        
        # Set task-specific number of classes
        if self.task == 'phenotype':
            self.num_classes = self.hparams.num_classes
        elif self.task == 'mortality':
            self.num_classes = 1
        elif self.task == 'los':
            self.num_classes = 7  # LoS has 7 classes (bins 2-8, excluding 0,1)
        else:
            raise ValueError(f"Unsupported task: {self.task}. Only 'mortality', 'phenotype', and 'los' are supported")
        
        self._init_model_components()

    def _init_model_components(self):
        """Initialize the Transformer encoder"""
        self.transformer_encoder = TransformerEncoder(
            input_size=self.hparams.input_dim,
            num_classes=self.num_classes,
            d_model=getattr(self.hparams, 'd_model', 256),
            n_head=getattr(self.hparams, 'n_head', 8),
            n_layers=getattr(self.hparams, 'n_layers', 2),
            dropout=getattr(self.hparams, 'dropout', 0.3),
            max_len=getattr(self.hparams, 'max_len', 500)
        )
        
        print(f"Transformer model initialized for {self.task} task")
        print(f"  - Input dimension: {self.hparams.input_dim}")
        print(f"  - Model dimension: {getattr(self.hparams, 'd_model', 256)}")
        print(f"  - Number of heads: {getattr(self.hparams, 'n_head', 8)}")
        print(f"  - Number of layers: {getattr(self.hparams, 'n_layers', 2)}")
        print(f"  - Dropout: {getattr(self.hparams, 'dropout', 0.3)}")
        print(f"  - Max length: {getattr(self.hparams, 'max_len', 500)}")
        print(f"  - Number of classes: {self.num_classes}")

    def forward(self, batch):
        """
        Forward pass for the Transformer model
        
        Args:
            batch: Dictionary containing 'ehr_ts', 'seq_len', and 'labels'
            
        Returns:
            Dictionary with 'loss', 'predictions', and 'labels'
        """
        # Extract inputs
        x = batch['ehr_ts']  # [batch_size, seq_len, input_dim]
        seq_lengths = batch['seq_len']  # [batch_size]
        labels = batch['labels']  # [batch_size, num_classes] or [batch_size]
        
        # Forward pass through Transformer encoder
        features, predictions = self.transformer_encoder(x, seq_lengths, output_prob=False)
        
        # Calculate loss
        loss = self.classification_loss(predictions, labels)
        
        return {
            'loss': loss,
            'predictions': torch.sigmoid(predictions),
            'labels': labels,
            'features': features
        }

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