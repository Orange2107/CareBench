import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import torchvision
import numpy as np
from collections import OrderedDict

# Import configurable encoders
from ..base.base_encoder import create_ehr_encoder, create_cxr_encoder

class FunctionalEncoder(nn.Module):
    def __init__(self, encoder, params, create_graph, mode, meta_train):
        super().__init__()
        self.encoder = encoder
        self.params = params
        self.create_graph = create_graph
        self.mode = mode
        self.meta_train = meta_train
        
        # Split parameters for each component
        self.cxr_params = params[:len(list(encoder.cxr_encoder.parameters()))]
        self.ehr_params = params[len(list(encoder.cxr_encoder.parameters())):len(list(encoder.cxr_encoder.parameters())) + len(list(encoder.ehr_encoder.parameters()))]
        self.fusion_params = params[len(list(encoder.cxr_encoder.parameters())) + len(list(encoder.ehr_encoder.parameters())):]
    
    def forward(self, batch: Dict[str, torch.Tensor], cxr_mean=None, mode=None, meta_train=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use the provided mode and meta_train if given, otherwise use the instance variables
        mode = mode if mode is not None else self.mode
        meta_train = meta_train if meta_train is not None else self.meta_train
        
        return self.encoder.forward(batch, mode=mode, cxr_mean=cxr_mean, meta_train=meta_train)
    
    def _functional_forward(self, module, params: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Helper function to perform forward pass with functional parameters."""
        # For configurable encoders, use their forward method directly
        if hasattr(module, 'forward'):
            return module(x)
        else:
            # Fallback for other modules
            return module(x)
    
    def _functional_lstm_forward(self, module, params: List[torch.Tensor], x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """Helper function to perform forward pass with functional parameters for LSTM-based encoders."""
        if hasattr(module, 'forward'):
            # For configurable encoders that return (prob, features)
            if hasattr(module, '__class__') and 'EHREncoder' in str(module.__class__):
                _, features = module(x, seq_lengths, output_prob=False)
                return features
            else:
                # For other modules
                _, features = module(x, seq_lengths)
                return features
        else:
            return module(x, seq_lengths)

class CXRInferNet(nn.Module):
    def __init__(self, ehr_dim, cxr_dim, output_layers=['default']):
        super(CXRInferNet, self).__init__()
        self.output_layers = output_layers
        
        # Mapping network from EHR features to CXR features
        self.fc1 = nn.Linear(ehr_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, cxr_dim)
        
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

    def _add_meta_train_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = self.softplus(torch.randn_like(x) + x)
        return len(output_layers) == len(outputs)

    def _add_meta_val_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = (x)
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None, meta_train=True):
        outputs = OrderedDict()
        
        if output_layers is None:
            output_layers = self.output_layers

        x = self.fc1(x)
        x = self.relu(x)
        if meta_train:
            if self._add_meta_train_output_and_check('fc1', x, outputs, output_layers):
                return outputs
        else:
            if self._add_meta_val_output_and_check('fc1', x, outputs, output_layers):
                return outputs

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        if meta_train:
            if self._add_meta_train_output_and_check('fc2', x, outputs, output_layers):
                return outputs
        else:
            if self._add_meta_val_output_and_check('fc2', x, outputs, output_layers):
                return outputs

        x = self.fc3(x)
        x = self.softplus(x)
        if meta_train:
            if self._add_meta_train_output_and_check('fc3', x, outputs, output_layers):
                return outputs
        else:
            if self._add_meta_val_output_and_check('fc3', x, outputs, output_layers):
                return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')

class SMILEncoder(nn.Module):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        
        # Set encoder defaults if not specified
        if not hasattr(hparams, 'ehr_encoder'):
            hparams['ehr_encoder'] = 'lstm'  # SMIL originally uses LSTM
        if not hasattr(hparams, 'cxr_encoder'):
            hparams['cxr_encoder'] = 'resnet50'
        if not hasattr(hparams, 'pretrained'):
            hparams['pretrained'] = True

        self._init_encoders(hparams)
        self._init_smil_components(hparams)

    def _init_encoders(self, hparams):
        """Initialize configurable encoders"""
        # Get encoder types
        ehr_encoder_type = hparams.get('ehr_encoder', 'lstm').lower()
        cxr_encoder_type = hparams.get('cxr_encoder', 'resnet50').lower()
        pretrained = hparams.get('pretrained', True)
        
        # Set task-specific parameters
        self.task = hparams.get('task', 'phenotype')
        if self.task == 'mortality':
            num_classes = 1
        else:
            num_classes = hparams.get('num_classes', 6)

        # ===== EHR Encoder Selection using Factory =====
        ehr_params = {
            'input_size': hparams.get('input_dim', 24),
            'num_classes': num_classes,
            'hidden_size': hparams.get('hidden_dim', 256),
            'dropout': hparams.get('dropout', 0.3),
        }
        
        # Add encoder-specific parameters
        if ehr_encoder_type == 'lstm':
            ehr_params.update({
                'num_layers': hparams.get('ehr_num_layers', hparams.get('layers', 2)),
                'bidirectional': hparams.get('ehr_bidirectional', True)
            })
        elif ehr_encoder_type == 'transformer':
            ehr_params.update({
                'd_model': ehr_params.pop('hidden_size'),  # transformer uses d_model
                'n_head': hparams.get('ehr_n_head', 8),
                'n_layers': hparams.get('ehr_n_layers', 2),
                'max_len': hparams.get('max_len', 500)
            })
        
        # Create EHR encoder
        self.ehr_encoder = create_ehr_encoder(encoder_type=ehr_encoder_type, **ehr_params)

        # ===== CXR Encoder Selection using Factory =====
        cxr_params = {
            'hidden_size': hparams.get('hidden_dim', 256),
            'pretrained': pretrained
        }
        
        # Create CXR encoder
        self.cxr_encoder = create_cxr_encoder(encoder_type=cxr_encoder_type, **cxr_params)

        print(f"SMIL model initialized:")
        print(f"  - EHR encoder: {ehr_encoder_type}, hidden_dim: {hparams.get('hidden_dim', 256)}")
        print(f"  - CXR encoder: {cxr_encoder_type}, hidden_dim: {hparams.get('hidden_dim', 256)}")
        print(f"  - Task: {self.task}, Pretrained: {pretrained}")

    def _init_smil_components(self, hparams):
        """Initialize SMIL specific components"""
        # Get actual feature dimensions from encoders by doing a dummy forward pass
        self._determine_feature_dimensions(hparams)
        
        # Set up task-specific parameters
        if self.task == 'mortality':
            num_classes = 1
        else:
            num_classes = hparams.get('num_classes', 6)
        
        # Set up fully connected layers
        self.fc1 = nn.Linear(self.total_feat_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Other components
        self.dropout = nn.Dropout(hparams.get('dropout', 0.3))
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
        # Add CXR reconstruction network
        self.cxr_infer_net = CXRInferNet(
            ehr_dim=self.ehr_feat_dim,
            cxr_dim=self.cxr_feat_dim
        )
        
        # For mean_projector, we need to determine the CXR k-means centers' feature dimension
        # This depends on the hidden_dim used when computing the k-means centers
        # Default is 256 for ResNet50, but we need to check the actual k-means centers
        kmeans_feat_dim = self._get_kmeans_feature_dim(hparams)
        
        # Create mean_projector that can handle the dimension mismatch
        if kmeans_feat_dim != self.cxr_feat_dim:
            # If dimensions don't match, create a projection from k-means dim to current CXR dim
            self.mean_projector = nn.Linear(kmeans_feat_dim, self.cxr_feat_dim)
            print(f"Warning: CXR k-means feature dim ({kmeans_feat_dim}) != current CXR feature dim ({self.cxr_feat_dim})")
            print(f"Created mean_projector: {kmeans_feat_dim} -> {self.cxr_feat_dim}")
        else:
            # If dimensions match, create identity mapping
            self.mean_projector = nn.Linear(self.cxr_feat_dim, self.cxr_feat_dim)
        
        self.weight_adapter = nn.Linear(self.cxr_feat_dim, 10)
        
        # Store k-means feature dimension for later use
        self.kmeans_feat_dim = kmeans_feat_dim
        
    def _get_kmeans_feature_dim(self, hparams):
        """Get the feature dimension of pre-computed CXR k-means centers"""
        # Default feature dimension for different encoders
        default_dims = {
            'resnet50': 256,  # Common default for ResNet50 after projection
        }
        
        cxr_encoder_type = hparams.get('cxr_encoder', 'resnet50').lower()
        
        # Try to infer from the k-means filename or use default
        # This is based on the naming convention used in compute_cxr_mean_kmeans.py
        kmeans_hidden_dim = hparams.get('kmeans_hidden_dim', None)
        
        if kmeans_hidden_dim is not None:
            return kmeans_hidden_dim
        else:
            # Use default dimension based on encoder type
            return default_dims.get(cxr_encoder_type, 256)

    def _determine_feature_dimensions(self, hparams):
        """Determine actual feature dimensions by doing dummy forward passes"""
        device = next(self.ehr_encoder.parameters()).device
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        input_dim = hparams.get('input_dim', 24)
        
        dummy_ehr = torch.randn(batch_size, seq_len, input_dim).to(device)
        dummy_seq_len = torch.tensor([seq_len, seq_len-1]).to(device)
        dummy_cxr = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # Get EHR feature dimension
        with torch.no_grad():
            ehr_output = self.ehr_encoder(dummy_ehr, dummy_seq_len, output_prob=False)
            if isinstance(ehr_output, tuple):
                ehr_features = ehr_output[1]  # Get features (not predictions)
            else:
                ehr_features = ehr_output
            
            # Flatten and get dimension
            ehr_features_flat = ehr_features.view(ehr_features.size(0), -1)
            self.ehr_feat_dim = ehr_features_flat.size(1)
            
            # Get CXR feature dimension
            cxr_features = self.cxr_encoder(dummy_cxr)
            cxr_features_flat = cxr_features.view(cxr_features.size(0), -1)
            self.cxr_feat_dim = cxr_features_flat.size(1)
            
            # Calculate total feature dimension
            self.total_feat_dim = self.ehr_feat_dim + self.cxr_feat_dim
            
            print(f"Feature dimensions determined:")
            print(f"  - EHR feature dim: {self.ehr_feat_dim}")
            print(f"  - CXR feature dim: {self.cxr_feat_dim}")
            print(f"  - Total feature dim: {self.total_feat_dim}")

    def _extract_features(self, batch):
        """Extract features using configurable encoders"""
        # Extract EHR features
        ehr_output = self.ehr_encoder(batch['ehr_ts'], batch['seq_len'], output_prob=False)
        if isinstance(ehr_output, tuple):
            ehr_features = ehr_output[1]  # Get features (not predictions)
        else:
            ehr_features = ehr_output
        ehr_features = ehr_features.view(ehr_features.size(0), -1)
        
        # Extract CXR features  
        cxr_features = self.cxr_encoder(batch['cxr_imgs'])
        cxr_features = cxr_features.view(cxr_features.size(0), -1)
        
        return ehr_features, cxr_features
    
    def forward(self, batch: Dict[str, torch.Tensor], cxr_mean=None, noise_layer=['fc0','fc1','fc2'], meta_train=True, mode='one') -> Dict[str, torch.Tensor]:
        if mode == 'one':
            assert cxr_mean is not None
            assert noise_layer is not None
            
            # Extract features using configurable encoders
            ehr_features, cxr_features = self._extract_features(batch)
            
            # Handle missing modalities
            if batch['has_cxr'].sum() < batch['has_cxr'].numel():  # If there are missing values
                missing_idx = (batch['has_cxr'] == 0).nonzero(as_tuple=True)[0]
                if len(missing_idx) > 0:
                    fc0 = ehr_features[missing_idx]
                    if 'fc0' in noise_layer:
                        # Get original weights (batch, cxr_feat_dim)
                        original_weight = self.cxr_infer_net(fc0, meta_train=meta_train)
                        # Use weight adapter to convert weights
                        weight = self.softplus(original_weight)
                        # Apply weight adapter and reshape to (batch, 10, 1)
                        adapted_weight = self.weight_adapter(weight).unsqueeze(-1)  # (batch, 10, 1)
                        projected_mean = self.mean_projector(cxr_mean.T).T  # (cxr_feat_dim, 10)
                        cxr_mean_expanded = projected_mean.unsqueeze(0).expand(len(missing_idx), -1, -1)  # (batch, cxr_feat_dim, 10)
                        reconstructed_cxr = torch.matmul(cxr_mean_expanded, adapted_weight)  # (batch, cxr_feat_dim, 1)
                        cxr_features[missing_idx] = reconstructed_cxr.squeeze(-1)
            
            # Fuse features
            combined_features = torch.cat([ehr_features, cxr_features], dim=1)
            
            # Forward through fully connected layers
            x = self.fc1(combined_features)
            x = self.relu(x)
            x = self.dropout(x)
            
            pred = self.fc2(x)
            pred = torch.sigmoid(pred)
            
            return pred, ehr_features, x, cxr_features
            
        elif mode == 'two':
            # Extract features using configurable encoders
            ehr_features, cxr_features = self._extract_features(batch)
            
            # Fuse features
            combined_features = torch.cat([ehr_features, cxr_features], dim=1)
            
            # Forward through fully connected layers
            x = self.fc1(combined_features)
            x = self.relu(x)
            x = self.dropout(x)
            
            pred = self.fc2(x)
            pred = torch.sigmoid(pred)
            
            return pred, ehr_features, x, cxr_features
        
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def functional(self, params, create_graph, mode='one', meta_train=True):
        return FunctionalEncoder(self, params, create_graph, mode, meta_train)

class SMILLoss(nn.Module):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        
        # Loss weights
        self.alpha = hparams.get('alpha', 0.05)  # Feature distillation weight
        self.beta = hparams.get('beta', 0.05)    # EHR mean distillation weight
        self.temperature = hparams.get('temperature', 3.0)  # Knowledge distillation temperature
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.task = hparams.get('task', 'phenotype')

    def meta_forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass for meta-training"""
        if self.task == 'los':
            # For LoS: use CrossEntropyLoss
            ce_loss = nn.CrossEntropyLoss(reduction='mean')
            return ce_loss(logits, labels.long().squeeze())
        else:
            # For binary/multi-label tasks: use BCELoss
            return self.bce_loss(logits.squeeze(), labels.float())
    
    def forward(self, mean_teacher: torch.Tensor, mean_student: torch.Tensor,
                map_teacher1: torch.Tensor, map_student1: torch.Tensor,
                map_teacher2: torch.Tensor, map_student2: torch.Tensor,
                pred_noise: torch.Tensor, pred_clean: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:

        # Classification loss for noisy prediction
        if self.task == 'los':
            # For LoS: use CrossEntropyLoss
            ce_loss = nn.CrossEntropyLoss(reduction='mean')
            loss_cls = ce_loss(pred_noise, label.long().squeeze())
        else:
            # For binary/multi-label tasks: use BCELoss
            loss_cls = self.bce_loss(pred_noise.squeeze(), label.float())
        
        # Knowledge distillation loss (teacher-student)
        loss_kd = self.mse_loss(pred_noise / self.temperature, pred_clean / self.temperature)
        
        # Feature distillation loss
        loss_feat = self.alpha * (
            self.mse_loss(map_student1, map_teacher1) + 
            self.mse_loss(map_student2, map_teacher2)
        )
        
        # Mean distillation loss
        loss_mean = self.beta * self.mse_loss(mean_student, mean_teacher)
        
        # Total loss
        total_loss = loss_cls + loss_kd + loss_feat + loss_mean
        
        return total_loss

# Legacy components for backward compatibility (if needed)
class LSTM(nn.Module):
    def __init__(self, input_dim=76, num_classes=1, hidden_dim=128, batch_first=True, dropout=0.0, layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        for layer in range(layers):
            setattr(self, f'layer{layer}', nn.LSTM(
                input_dim, hidden_dim,
                batch_first=batch_first,
                dropout=dropout)
            )
            input_dim = hidden_dim
        self.do = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        self.feats_dim = hidden_dim
        self.dense_layer = nn.Linear(hidden_dim, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        for model in self.modules():
            if type(model) in [nn.Linear]:
                nn.init.xavier_uniform_(model.weight)
                nn.init.zeros_(model.bias)
            elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x, seq_lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f'layer{layer}')(x)
        feats = ht.squeeze()
        if self.do is not None:
            feats = self.do(feats)
        out = self.dense_layer(feats)
        scores = torch.sigmoid(out)
        return scores, feats

class CXRModels(nn.Module):
    def __init__(self, args, device='cpu'):
        super(CXRModels, self).__init__()
        self.args = args
        self.device = device
        self.vision_backbone = getattr(torchvision.models, self.args.get('vision_backbone', 'resnet50'))(pretrained=self.args.get('pretrained', True))
        classifiers = ['classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
            break
        self.bce_loss = torch.nn.BCELoss(size_average=True)
        self.classifier = nn.Sequential(nn.Linear(d_visual, self.args.get('num_classes', 1)))
        self.feats_dim = d_visual

    def forward(self, x, labels=None, n_crops=0, bs=16):
        visual_feats = self.vision_backbone(x)
        preds = self.classifier(visual_feats)
        preds = torch.sigmoid(preds)

        if n_crops > 0:
            preds = preds.view(bs, n_crops, -1).mean(1)
        
        lossvalue_bce = torch.zeros(1).to(self.device)
        if labels is not None:
            lossvalue_bce = self.bce_loss(preds, labels)

        return preds, lossvalue_bce, visual_feats
