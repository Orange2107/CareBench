import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights

import torchxrayvision as xrv


class EHREncoderFactory:
    @staticmethod
    def create_encoder(encoder_type, **kwargs):
        if encoder_type.lower() == 'lstm':
            return LSTMEncoder(**kwargs)
        elif encoder_type.lower() == 'transformer':
            return TransformerEncoder(**kwargs)
        else:
            raise ValueError(f"Unsupported EHR encoder type: {encoder_type}. Supported types: 'lstm', 'transformer'")

class CXREncoderFactory:
    @staticmethod
    def create_encoder(encoder_type, hidden_size=256, pretrained=True, **kwargs):
        if encoder_type.lower() == 'resnet50':
            return ResNet50Encoder(hidden_size=hidden_size, pretrained=pretrained)
        elif encoder_type.lower() == 'densenet121-res224-chex':
            return DenseNet121CheXEncoder(hidden_size=hidden_size, pretrained=pretrained)
        elif encoder_type.lower() == 'densenet121-imagenet':
            return DenseNet121ImageNetEncoder(hidden_size=hidden_size, pretrained=pretrained)
        else:
            raise ValueError(
                f"Unsupported CXR encoder type: {encoder_type}. "
                f"Supported types: 'resnet50', 'densenet121-res224-chex', 'densenet121-imagenet'"
            )

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.rand(1, max_len, d_model))
        self.pe.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, num_layers=2, 
                 dropout=0.3, bidirectional=True, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.feature_projection = nn.Linear(lstm_output_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, seq_lengths, output_prob=True):
        x = self.input_projection(x)
        if isinstance(seq_lengths, list):
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=x.device)
        elif not isinstance(seq_lengths, torch.Tensor):
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=x.device)
        elif seq_lengths.device != x.device:
            seq_lengths = seq_lengths.to(x.device)
        seq_lengths = torch.clamp(seq_lengths, min=1, max=x.size(1))
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_x)
        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            lstm_feat = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            lstm_feat = hidden[-1]
        lstm_feat = self.dropout(lstm_feat)
        feat = self.feature_projection(lstm_feat)
        feat = self.dropout(feat)
        prediction = self.classifier(feat)
        if output_prob:
            prediction = prediction.sigmoid()
        return feat, prediction
    
    def get_output_dim(self):
        return self.hidden_size

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_classes, d_model=256, n_head=8, n_layers=2,
                 dropout=0.3, max_len=500, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            batch_first=True, 
            dropout=dropout,
            dim_feedforward=d_model * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, seq_lengths, output_prob=True):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        if isinstance(seq_lengths, list):
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=x.device)
        elif not isinstance(seq_lengths, torch.Tensor):
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=x.device)
        elif seq_lengths.device != x.device:
            seq_lengths = seq_lengths.to(x.device)
        seq_lengths = torch.clamp(seq_lengths, min=1, max=x.size(1))
        attn_mask = torch.stack([
            torch.cat([
                torch.zeros(len_, device=x.device),
                torch.ones(max(seq_lengths) - len_, dtype=torch.bool, device=x.device)
            ])
            for len_ in seq_lengths
        ])
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        padding_mask = torch.ones_like(attn_mask).unsqueeze(2)
        padding_mask[attn_mask == float('-inf')] = 0
        feat = (padding_mask * transformer_out).sum(dim=1) / padding_mask.sum(dim=1)
        feat = self.dropout(feat)
        prediction = self.classifier(feat)
        if output_prob:
            prediction = prediction.sigmoid()
        return feat, prediction
    
    def get_output_dim(self):
        return self.d_model

class ResNet50Encoder(nn.Module):
    def __init__(self, hidden_size=256, pretrained=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=hidden_size)
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_output_dim(self):
        return self.hidden_size

class DenseNet121CheXEncoder(nn.Module):
    """DenseNet121 encoder using TorchXRayVision densenet121-res224-chex weights"""
    def __init__(self, hidden_size=256, pretrained=True):
        super().__init__()
        self.hidden_size = hidden_size
        
        if pretrained:
            # Load pretrained DenseNet121 model from TorchXRayVision
            # TorchXRayVision DenseNet is trained on single-channel (grayscale) images
            self.backbone = xrv.models.DenseNet(in_channels=1, weights="densenet121-res224-chex")
        else:
            # Load DenseNet121 without pretrained weights
            self.backbone = xrv.models.DenseNet(in_channels=1, weights=None)
        
        # DenseNet121 outputs 1024 features before classifier
        feature_dim = 1024
        # Replace classifier with projection to hidden_size, same as ImageNet version
        self.backbone.classifier = nn.Linear(feature_dim, hidden_size)
        
    def forward(self, x):
        # TorchXRayVision DenseNet expects single-channel (grayscale) input
        # Convert RGB (3 channels) to grayscale (1 channel) if needed
        if x.shape[1] == 3:
            # Use standard RGB to grayscale conversion weights
            # 0.299*R + 0.587*G + 0.114*B
            x = (0.299 * x[:, 0:1, :, :] + 
                 0.587 * x[:, 1:2, :, :] + 
                 0.114 * x[:, 2:3, :, :])
        
        # TorchXRayVision expects input in range [-1024, 1024] (DICOM range)
        # But input is ImageNet normalized (range ~[-2, 2])
        # Convert from ImageNet normalization back to [0, 1] range, then scale to [-1024, 1024]
        # ImageNet normalization: (x - mean) / std, where mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # For grayscale, approximate mean=0.45, std=0.225
        # Reverse: x_original = x_normalized * std + mean
        # Then scale to DICOM range: x_dicom = (x_original - 0.5) * 2048
        mean_gray = 0.45  # Approximate grayscale mean
        std_gray = 0.225  # Approximate grayscale std
        x = (x * std_gray + mean_gray)  # Denormalize to [0, 1]
        x = (x - 0.5) * 2048  # Scale to [-1024, 1024] range
        
        # Use features method to bypass forward's op_norm that expects 18 classes
        # This is necessary because we replaced the classifier with our own projection
        if hasattr(self.backbone, 'features'):
            features = self.backbone.features(x)
            if len(features.shape) > 2:
                features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            return self.backbone.classifier(features)
        else:
            raise NotImplementedError("TorchXRayVision DenseNet model structure not recognized")
    
    def get_output_dim(self):
        return self.hidden_size

class DenseNet121ImageNetEncoder(nn.Module):
    """DenseNet121 encoder using torchvision ImageNet weights"""
    def __init__(self, hidden_size=256, pretrained=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        # DenseNet121 from torchvision outputs 1024 features before classifier
        feature_dim = 1024
        self.backbone.classifier = nn.Linear(feature_dim, hidden_size)
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_output_dim(self):
        return self.hidden_size

def create_ehr_encoder(encoder_type, **kwargs):
    return EHREncoderFactory.create_encoder(encoder_type, **kwargs)

def create_cxr_encoder(encoder_type, **kwargs):
    return CXREncoderFactory.create_encoder(encoder_type, **kwargs)