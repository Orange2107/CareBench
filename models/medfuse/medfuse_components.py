import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import ResNet
from .ehr_encoder import DisentangledEHRTransformer


class Fusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        target_classes = self.args.num_classes
        if self.args.drfuse_encoder:
            lstm_in = self.ehr_model.d_model
            lstm_out = self.ehr_model.d_model
            projection_in = self.ehr_model.d_model
            feats_dim = 2 * self.ehr_model.d_model
        else:
            lstm_in = self.ehr_model.feats_dim
            lstm_out = self.ehr_model.feats_dim
            projection_in = self.cxr_model.feats_dim
            feats_dim = 2 * self.ehr_model.feats_dim

        if self.args.labels_set == 'radiology':
            target_classes = self.args.vision_num_classes
            lstm_in = self.cxr_model.feats_dim
            projection_in = self.ehr_model.feats_dim
            feats_dim = 2 * self.ehr_model.feats_dim

        self.projection = nn.Linear(projection_in, lstm_in)
        
        
        self.fused_cls = nn.Linear(feats_dim, self.args.num_classes)

        self.align_loss = CosineLoss()
        self.kl_loss = KLDivLoss()

        self.lstm_fused_cls = nn.Linear(lstm_out, target_classes)

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout = 0.0)
            
    def forward_uni_cxr(self, x, seq_lengths=None, img=None):
        cxr_preds, _, feats = self.cxr_model(img)
        return {
            'uni_cxr': cxr_preds,
            'cxr_feats': feats
        }
    
    def forward(self, x, seq_lengths=None, img=None, pairs=None):
        if self.args.fusion_type == 'uni_cxr':
            return self.forward_uni_cxr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type in ['joint', 'early', 'late_avg', 'unified']:
            return self.forward_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs)
        elif self.args.fusion_type == 'uni_ehr':
            return self.forward_uni_ehr(x, seq_lengths=seq_lengths, img=img)
        elif self.args.fusion_type == 'lstm':
            return self.forward_lstm_fused(x, seq_lengths=seq_lengths, img=img, pairs=pairs)
        elif self.args.fusion_type == 'uni_ehr_lstm':
            return self.forward_lstm_ehr(x, seq_lengths=seq_lengths, img=img, pairs=pairs)

    def forward_uni_ehr(self, x, seq_lengths=None, img=None):
        ehr_preds, feats = self.ehr_model(x, seq_lengths)
        return {
            'uni_ehr': ehr_preds,
            'ehr_feats': feats
        }

    def forward_fused(self, x, seq_lengths=None, img=None, pairs=None):
        ehr_preds, ehr_feats = self.ehr_model(x, seq_lengths)
        cxr_preds, _, cxr_feats = self.cxr_model(img)
        projected = self.projection(cxr_feats)

        loss = self.align_loss(projected, ehr_feats)

        feats = torch.cat([ehr_feats, projected], dim=1)
        fused_preds = self.fused_cls(feats)

        late_avg = (cxr_preds + ehr_preds)/2
        
        return {
            'early': fused_preds, 
            'joint': fused_preds, 
            'late_avg': late_avg,
            'align_loss': loss,
            'ehr_feats': ehr_feats,
            'cxr_feats': projected,
            'unified': fused_preds
        }
    
    def forward_lstm_fused(self, x, seq_lengths=None, img=None, pairs=None):
        if self.args.labels_set == 'radiology':
            _, ehr_feats = self.ehr_model(x, seq_lengths)
            
            _, _, cxr_feats = self.cxr_model(img)
            feats = cxr_feats.unsqueeze(1)
            ehr_feats = self.projection(ehr_feats)
            
            mask = ~pairs.bool()
            ehr_feats[mask] = 0
            
            feats = torch.cat([feats, ehr_feats.unsqueeze(1)], dim=1)
        else:
            # original MedFuse Encoder
            # _, ehr_feats = self.ehr_model(x, seq_lengths) # [batch, 256]
            # _, _, cxr_feats = self.cxr_model(img) # [batch,512]
            # cxr_feats = self.projection(cxr_feats)  # [batch, 256]


            # Drfuse Encoder shape [batch,256]
            if self.args.drfuse_encoder:
                print("Drfuse Encoder")
                ehr_feats,_ = self.ehr_model(x, seq_lengths)
                cxr_feats = self.cxr_model(img)
            else:
                _, ehr_feats = self.ehr_model(x, seq_lengths) # [batch, 256]
                _, _, cxr_feats = self.cxr_model(img) # [batch,512]
                cxr_feats = self.projection(cxr_feats)  # [batch, 256]

            mask = ~pairs.bool()
            cxr_feats[mask] = 0
            
            if len(ehr_feats.shape) == 1:
                feats = ehr_feats.unsqueeze(0).unsqueeze(0)
            else:
                feats = ehr_feats.unsqueeze(1) # [batch,1,256]
            
            feats = torch.cat([feats, cxr_feats.unsqueeze(1)], dim=1)
        
        new_seq_lengths = torch.ones(len(seq_lengths), dtype=torch.long, device=pairs.device)
        
        if not isinstance(pairs, torch.Tensor) or pairs.dtype != torch.bool:
            pairs_bool = pairs.bool() if isinstance(pairs, torch.Tensor) else torch.tensor(pairs, dtype=torch.bool, device=new_seq_lengths.device)
        else:
            pairs_bool = pairs

        if pairs_bool.device != new_seq_lengths.device:
            pairs_bool = pairs_bool.to(new_seq_lengths.device)

        new_seq_lengths[pairs_bool] = 2
        
        feats = torch.nn.utils.rnn.pack_padded_sequence(
            feats, new_seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        x, (ht, _) = self.lstm_fusion_layer(feats)
        out = ht.squeeze()
        fused_preds = self.lstm_fused_cls(out)
        
        return {
            'lstm': fused_preds,
            'ehr_feats': ehr_feats,
            'cxr_feats': cxr_feats,
        }
    
    def forward_lstm_ehr(self, x, seq_lengths=None, img=None, pairs=None):
        _, ehr_feats = self.ehr_model(x, seq_lengths)
        feats = ehr_feats.unsqueeze(1)
        
        new_seq_lengths = torch.ones(len(seq_lengths), dtype=torch.long, device=ehr_feats.device)
        
        feats = torch.nn.utils.rnn.pack_padded_sequence(
            feats, new_seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        x, (ht, _) = self.lstm_fusion_layer(feats)
        out = ht.squeeze()
        fused_preds = self.lstm_fused_cls(out)
        
        return {
            'uni_ehr_lstm': fused_preds,
        }
    

class LSTM(nn.Module):

    def __init__(self, input_dim=76, num_classes=1, hidden_dim=128, batch_first=True, dropout=0.0, layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        for layer in range(layers):
            setattr(self, f'layer{layer}', nn.LSTM(
                input_dim, hidden_dim,
                batch_first=batch_first,
                dropout = dropout)
            )
            input_dim = hidden_dim
        self.do = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        self.feats_dim = hidden_dim
        self.dense_layer = nn.Linear(hidden_dim, num_classes)
        self.initialize_weights()
        # self.activation = torch.sigmoid
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
        classifiers = [ 'classifier', 'fc']
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
    
class KLDivLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(KLDivLoss, self).__init__()

        self.temperature = temperature
    def forward(self, emb1, emb2):
        emb1 = softmax(emb1/self.temperature, dim=1).detach()
        emb2 = log_softmax(emb2/self.temperature, dim=1)
        loss_kldiv = kl_div(emb2, emb1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        loss_kldiv = torch.mean(loss_kldiv)
        return loss_kldiv
 
class RankingLoss(nn.Module):
    def __init__(self, neg_penalty=0.03):
        super(RankingLoss, self).__init__()

        self.neg_penalty = neg_penalty
    def forward(self, ranks, labels, class_ids_loaded, device):
        '''
        for each correct it should be higher then the absence 
        '''
        labels = labels[:, class_ids_loaded]
        ranks_loaded = ranks[:, class_ids_loaded]
        neg_labels = 1+(labels*-1)
        loss_rank = torch.zeros(1).to(device)
        for i in range(len(labels)):
            correct = ranks_loaded[i, labels[i]==1]
            wrong = ranks_loaded[i, neg_labels[i]==1]
            correct = correct.reshape((-1, 1)).repeat((1, len(wrong)))
            wrong = wrong.repeat(len(correct)).reshape(len(correct), -1)
            image_level_penalty = ((self.neg_penalty+wrong) - correct)
            image_level_penalty[image_level_penalty<0]=0
            loss_rank += image_level_penalty.sum()
        loss_rank /=len(labels)

        return loss_rank
    
class CosineLoss(nn.Module):
    
    def forward(self, cxr, ehr ):
        a_norm = ehr / ehr.norm(dim=1)[:, None]
        b_norm = cxr / cxr.norm(dim=1)[:, None]
        loss = 1 - torch.mean(torch.diagonal(torch.mm(a_norm, b_norm.t()), 0))
        
        return loss