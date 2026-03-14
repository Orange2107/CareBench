import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.base_encoder import create_cxr_encoder

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
        self.vision_backbone = getattr(torchvision.models, self.args.vision_backbone)(pretrained=self.args.pretrained)
        classifiers = ['classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
            break
        self.bce_loss = torch.nn.BCELoss(size_average=True)
        self.classifier = nn.Sequential(nn.Linear(d_visual, self.args.vision_num_classes))
        self.feats_dim = d_visual

    def forward(self, x, labels=None, n_crops=0, bs=16):
        lossvalue_bce = torch.zeros(1).to(self.device)
        visual_feats = self.vision_backbone(x)
        preds = self.classifier(visual_feats)
        preds = torch.sigmoid(preds)
        if n_crops > 0:
            preds = preds.view(bs, n_crops, -1).mean(1)
        if labels is not None:
            lossvalue_bce = self.bce_loss(preds, labels)
        return preds, lossvalue_bce, visual_feats

class MultiModalTransformerSharedEncoder(nn.Module):
    def __init__(
        self,
        feat_dim=256,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        ehr_input_dim=498,
        max_seq_len=500,
        cxr_encoder: str = 'resnet50',
        pretrained: bool = True,
        hf_model_id: str = 'codewithdark/vit-chest-xray',
        freeze_vit: bool = True,
        bias_tune: bool = False,
        partial_layers: int = 0,
    ):
        super(MultiModalTransformerSharedEncoder, self).__init__()
        self.feat_dim = feat_dim
        self.max_seq_len = max_seq_len
        assert feat_dim % nhead == 0, f"Feature dimension ({feat_dim}) must be divisible by nhead ({nhead})"
        self.ehr_projection = nn.Linear(ehr_input_dim, feat_dim)

        self.cxr_backbone = create_cxr_encoder(
            encoder_type=cxr_encoder,
            hidden_size=feat_dim,
            pretrained=pretrained,
            hf_model_id=hf_model_id,
            freeze_vit=freeze_vit,
            bias_tune=bias_tune,
            partial_layers=partial_layers,
        )
        self.shared_cxr_projection = nn.Identity()
        self.modality_embedding = nn.Parameter(torch.zeros(2, feat_dim))
        nn.init.normal_(self.modality_embedding, mean=0, std=0.02)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, feat_dim))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(feat_dim, feat_dim)

    def forward(self, ehr_data, cxr_data, seq_lengths, valid_cxr):
        batch_size = ehr_data.size(0)
        device = ehr_data.device
        ehr_projected = self.ehr_projection(ehr_data)
        ehr_modality_emb = self.modality_embedding[0].unsqueeze(0).unsqueeze(0)
        ehr_projected = ehr_projected + ehr_modality_emb
        seq_len = ehr_projected.size(1)
        ehr_projected = ehr_projected + self.pos_encoder[:, :seq_len, :]
        ehr_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        for i, length in enumerate(seq_lengths):
            if length < seq_len:
                ehr_mask[i, length:] = True
        cxr_features = self.cxr_backbone(cxr_data)
        cxr_ft = self.shared_cxr_projection(cxr_features)
        cxr_modality_emb = self.modality_embedding[1].unsqueeze(0)
        cxr_ft = cxr_ft + cxr_modality_emb
        cxr_ft = cxr_ft.unsqueeze(1)
        cxr_ft = cxr_ft + self.pos_encoder[:, :1, :]
        valid_cxr_bool = valid_cxr.bool() if valid_cxr.dtype != torch.bool else valid_cxr
        cxr_mask = ~valid_cxr_bool.unsqueeze(1)
        max_total_len = seq_len + 1
        combined_input = torch.zeros(batch_size * 2, max_total_len, self.feat_dim, device=device)
        combined_mask = torch.ones(batch_size * 2, max_total_len, device=device, dtype=torch.bool)
        for i in range(batch_size):
            ehr_idx = i * 2
            combined_input[ehr_idx, :seq_len] = ehr_projected[i]
            combined_mask[ehr_idx, :seq_lengths[i]] = False
            cxr_idx = i * 2 + 1
            combined_input[cxr_idx, 0] = cxr_ft[i, 0]
            combined_mask[cxr_idx, 0] = cxr_mask[i, 0]
        shared_features = []
        for i in range(batch_size * 2):
            valid_mask = ~combined_mask[i]
            valid_len = valid_mask.sum().item()
            if valid_len > 0:
                valid_input = combined_input[i, :valid_len].unsqueeze(0)
                transformed = self.transformer(valid_input)
                pooled = transformed.mean(dim=1)
                shared_features.append(pooled)
            else:
                shared_features.append(torch.zeros(1, self.feat_dim, device=device))
        shared_ft = torch.cat(shared_features, dim=0)
        shared_ft = self.output_projection(shared_ft)
        shared_gft = self.global_pool(shared_ft.unsqueeze(-1)).squeeze(-1)
        return shared_ft, shared_gft

class CompositionalLayer(nn.Module):
    def __init__(self, feat_dim=256, weight_std=False, normalization_sign=False):
        super().__init__()
        self.normalization = normalization_sign
        self.conv = nn.Linear(feat_dim * 2, feat_dim)
    
    def forward(self, f1, f2):
        if self.normalization:
            f1_n = F.normalize(f1, dim=1)
            f2_n = F.normalize(f2, dim=1)
            residual = torch.cat((f1_n, f2_n), 1)
        else:
            residual = torch.cat((f1, f2), 1)
        residual = self.conv(residual)
        features = f1 + residual
        return features

class FusionClassifier(nn.Module):
    def __init__(self, feat_dim=256, num_classes=1, dropout=0.2):
        super(FusionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)
