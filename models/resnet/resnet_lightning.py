import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from ..base.base_encoder import ResNet50Encoder

@ModelRegistry.register('resnet')
class ResNetModel(BaseFuseTrainer):
    """
    Single-modal ResNet model for CXR image classification
    - Uses ResNet50Encoder from base encoder
    - Supports both mortality and phenotype prediction tasks
    - Inherits training/validation/test steps from BaseFuseTrainer
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
        self.resnet_encoder = ResNet50Encoder(
            hidden_size=getattr(self.hparams, 'hidden_size', 256),
            pretrained=getattr(self.hparams, 'pretrained', True)
        )
        self.classifier = nn.Linear(
            self.resnet_encoder.get_output_dim(),
            self.num_classes
        )
        self.dropout = nn.Dropout(getattr(self.hparams, 'dropout', 0.3))

        print(f"ResNet model initialized for {self.task} task")
        print(f"  - Hidden size: {getattr(self.hparams, 'hidden_size', 256)}")
        print(f"  - Pretrained: {getattr(self.hparams, 'pretrained', True)}")
        print(f"  - Dropout: {getattr(self.hparams, 'dropout', 0.3)}")
        print(f"  - Number of classes: {self.num_classes}")

    def forward(self, batch):
        images = batch['cxr_imgs']
        labels = batch['labels']
        features = self.resnet_encoder(images)
        features = self.dropout(features)
        predictions = self.classifier(features)
        predictions_prob = torch.sigmoid(predictions)
        loss = self.classification_loss(predictions, labels)

        return {
            'loss': loss,
            'predictions': predictions_prob,
            'labels': labels,
            'features': features
        }

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