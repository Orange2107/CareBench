import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .moe_module import *
from itertools import combinations
import torchvision

class FlexMoEModel(nn.Module):
    """
    FlexMoE model for multimodal clinical data.
    This is a wrapper around the FlexMoE implementation that adapts it to 
    handle modality-specific data processing for the medical benchmark.

    Args:
        num_modalities (int): Number of modalities (typically 2 for EHR and CXR)
        full_modality_index (int): Index of the combination with all modalities
        input_dims (list): List of input dimensions for each modality
        hidden_dim (int): Hidden dimension size
        output_dim (int): Number of output classes
        num_layers (int): Number of transformer layers
        num_layers_pred (int): Number of prediction layers
        num_experts (int): Number of experts in the MoE
        num_routers (int): Number of routers
        top_k (int): Number of experts to route to
        num_heads (int): Number of attention heads in transformer
        dropout (float): Dropout probability
        num_patches (int): Number of patches for input token
    """
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
        vision_backbone='resnet50',
        pretrained=True
    ):
        super(FlexMoEModel, self).__init__()
        
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        
        self.modality_encoders = nn.ModuleList()
        
        self.modality_encoders.append(EHRPatchEmbeddings(
            feature_size=input_dims[0],
            num_patches=self.num_patches,
            embed_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers
        ))
        
        if len(input_dims) > 1:
            self.modality_encoders.append(CXREncoder(
                hidden_dim=hidden_dim,
                num_patches=self.num_patches,
                vision_backbone=vision_backbone,
                pretrained=pretrained
            ))
        
        self.flexmoe = FlexMoE(
            num_modalities=num_modalities,
            full_modality_index=full_modality_index,
            num_patches=self.num_patches,
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
        """
        Forward pass through the FlexMoE model.
        Args:
            inputs (list): List of input tensors for each modality
            expert_indices (torch.Tensor): Expert indices for routing
            has_modality (list): List of boolean tensors indicating modality availability
        """
        processed_inputs = []
        batch_size = inputs[0].size(0)
        
        for i, x in enumerate(inputs):
            encoded = torch.zeros(
                (batch_size, self.num_patches, self.hidden_dim),
                device=x.device if x is not None else inputs[0].device
            )
            
            if x is not None:
                if has_modality is not None and has_modality[i] is not None:
                    mask = has_modality[i].bool()
                    if mask.any():
                        encoded[mask] = self.modality_encoders[i](x[mask])
                    if (~mask).any():
                        encoded[~mask] = self.missing_embeds[expert_indices[~mask].long(), i]
                else:
                    encoded = self.modality_encoders[i](x)
            else:
                encoded = self.missing_embeds[expert_indices.long(), i]
            
            processed_inputs.append(encoded)
        
        return self.flexmoe(*processed_inputs, expert_indices=expert_indices)
    
    def gate_loss(self):
        return self.flexmoe.gate_loss()
    
    def assign_expert(self, combination):
        return self.flexmoe.assign_expert(combination)
    
    def set_full_modality(self, is_full_modality):
        self.flexmoe.set_full_modality(is_full_modality)

class FlexMoE(nn.Module):
    def __init__(self, num_modalities, full_modality_index, num_patches, hidden_dim, output_dim, num_layers, num_layers_pred, num_experts, num_routers, top_k, num_heads=2, dropout=0.5):
        super(FlexMoE, self).__init__()
        layers = []
        _sparse = True
        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=_sparse, full_modality_index=full_modality_index, top_k=top_k))
        for j in range(num_layers - 1):
            _sparse = not _sparse
            layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=_sparse, full_modality_index=full_modality_index, top_k=top_k))
        layers.append(MLP(hidden_dim*num_modalities, hidden_dim, output_dim, num_layers_pred, activation=nn.ReLU(), dropout=0.5))
        
        self.network = nn.Sequential(*layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, np.sum([num_patches]*num_modalities), hidden_dim))
        self.combination_to_index = self._create_combination_index(num_modalities)

    def forward(self, *inputs, expert_indices=None, is_full_modality=None):
        chunk_size = [input.shape[1] for input in inputs]
        x = torch.cat(inputs, dim=1)
        
        if self.pos_embed is not None:
            x += self.pos_embed

        x = torch.split(x, chunk_size, dim=1)

        for i in range(len(self.network) - 1):
            if expert_indices is not None and hasattr(self.network[i], 'set_expert_index'):
                self.network[i].set_expert_index(expert_indices)
            x = self.network[i](x)
        x = [item.mean(dim=1) for item in x]
        x = torch.cat(x, dim=1)
        x = self.network[-1](x)
        return x

    def gate_loss(self):
        g_loss = []
        for mn, mm in self.named_modules():
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is None:
                        pass
                    else:
                        g_loss.append(i_loss)
        return sum(g_loss)

    def _create_combination_index(self, num_modalities):
        combinations_list = []
        for r in range(1, num_modalities + 1):
            combinations_list.extend(combinations(range(num_modalities), r))
        combination_to_index = {tuple(sorted(comb)): idx for idx, comb in enumerate(combinations_list)}
        return combination_to_index

    def assign_expert(self, combination):
        index = self.combination_to_index.get(tuple(sorted(combination)))
        return index

    def set_full_modality(self, is_full_modality):
        for layer in self.network:
            if hasattr(layer, 'set_full_modality'):
                layer.set_full_modality(is_full_modality)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Custom3DCNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super(Custom3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.dropout1 = nn.Dropout3d(0.2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=3)
        self.dropout2 = nn.Dropout3d(0.2)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, hidden_dim, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.dropout3 = nn.Dropout3d(0.2)

        self.fc = nn.Linear(hidden_dim * 3 * 3 * 4, hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.pool3(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        pad_size = num_patches*patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(x.shape[0], self.num_patches, self.patch_size)
        x = self.projection(x)
        return x
    
class EHRPatchEmbeddings(nn.Module):
    """
    EHR to Patch Embedding.
    Input shape: [batch_size, seq_len, features]
    Output shape: [batch_size, num_patches, embed_dim]
    """
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25, num_layers=2):
        super().__init__()
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.embed_dim = embed_dim
        
        self.seq_projection = nn.Linear(feature_size, embed_dim)
        
        self.patch_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,  
            batch_first=True,
            bidirectional=True
        )
        
        self.patch_attention = nn.MultiheadAttention(
            embed_dim=embed_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.patch_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.final_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        x = self.seq_projection(x)
        
        if seq_len % self.num_patches != 0:
            pad_size = self.num_patches - (seq_len % self.num_patches)
            x = F.pad(x, (0, 0, 0, pad_size))
            seq_len = seq_len + pad_size
        
        x = x.view(batch_size, self.num_patches, -1, self.embed_dim)
        x = x.view(batch_size * self.num_patches, -1, self.embed_dim)
        lstm_out, _ = self.patch_lstm(x)
        attn_out, _ = self.patch_attention(lstm_out, lstm_out, lstm_out)
        x = self.patch_projection(attn_out)
        x = x[:, -1, :]
        x = x.view(batch_size, self.num_patches, self.embed_dim)
        x = self.final_projection(x)
        
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                num_experts,
                num_routers,
                d_model, 
                num_head, 
                dropout=0.1, 
                activation=nn.GELU, 
                hidden_times=2, 
                mlp_sparse = False, 
                self_attn = True,
                full_modality_index=4,
                top_k=2,
                **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(
            d_model, num_heads=num_head, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)
        
        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn
        self.expert_index = None
        self.full_modality_index = full_modality_index

        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(num_expert=num_experts, n_router=num_routers, d_model=d_model, d_hidden=d_model * hidden_times, activation=nn.GELU(), top_k=top_k, **kwargs)
        else:
            self.mlp = MLP(input_dim=d_model, hidden_dim=d_model * hidden_times, output_dim=d_model, num_layers=2, activation=nn.GELU(), dropout=dropout)

    def forward(self, x, attn_mask = None):
        if self.self_attn:
            chunk_size = [item.shape[1] for item in x]
            x = self.norm1(torch.cat(x, dim=1))
            kv = x
            x = self.attn(x, kv, attn_mask)
            x = x + self.dropout1(x)
            x = torch.split(x, chunk_size, dim=1)
            x = [item for item in x]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        else:
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m, attn_mask)
            x = [x[i]+self.dropout1(x[i]) for i in range(len(chunk_size))]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        return x

    def set_expert_index(self, expert_index):
        self.expert_index = expert_index

    def set_full_modality(self, is_full_modality):
        if hasattr(self.mlp, 'set_full_modality'):
            self.mlp.set_full_modality(is_full_modality)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv, attn_mask=None):
        eps = 1e-6

        Bx, Nx, Cx = x.shape
        B, N, C = kv.shape
        q = self.q(x).reshape(Bx, Nx, self.num_heads, Cx//self.num_heads)
        q = q.permute(0, 2, 1, 3)
        kv = self.kv(kv)
        kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(Bx, Nx, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CXREncoder(nn.Module):
    def __init__(self, hidden_dim, num_patches, vision_backbone='resnet50', pretrained=True):
        super().__init__()
        self.vision_backbone = getattr(torchvision.models, vision_backbone)(pretrained=pretrained)
        
        self.backbone = nn.Sequential(*list(self.vision_backbone.children())[:-2])
        
        if vision_backbone.startswith('resnet'):
            if vision_backbone in ['resnet18', 'resnet34']:
                self.feature_dim = 512
            else:
                self.feature_dim = 2048
        elif vision_backbone.startswith('densenet'):
            self.feature_dim = 1024
        elif vision_backbone.startswith('efficientnet'):
            self.feature_dim = 1280
        else:
            self.feature_dim = 2048
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((int(math.sqrt(num_patches)), int(math.sqrt(num_patches))))
        
        self.projection = nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1)
        
        self.fc = nn.Linear(hidden_dim * num_patches, hidden_dim * num_patches)
        
        self.patch_embed = PatchEmbeddings(
            feature_size=hidden_dim * num_patches,
            num_patches=num_patches,
            embed_dim=hidden_dim
        )
        
    def forward(self, x):
        x = self.backbone(x)
        
        x = self.adaptive_pool(x)
        
        x = self.projection(x)
        
        B, C, H, W = x.shape
        x = x.view(B, -1)
        
        x = self.fc(x)
        
        x = self.patch_embed(x)
        
        return x