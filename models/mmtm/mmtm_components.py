# models/MMTM/mmtm_components.py
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
import torch.nn.functional as F
from torch import Tensor

class MMTMLayer(nn.Module):
    """Multimodal Transfer Module (MMTM)层实现"""
    def __init__(self, dim_visual, dim_ehr, ratio):
        super(MMTMLayer, self).__init__()
        dim = dim_visual + dim_ehr
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)
        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_ehr)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, visual, skeleton):
        squeeze_array = []
        visual_view = visual.view(visual.shape[:2] + (-1,))
        squeeze_array.append(torch.mean(visual_view, dim=-1))
        ehr_avg = torch.mean(skeleton, dim=1)
        squeeze_array.append(ehr_avg)
        squeeze = torch.cat(squeeze_array, 1)
        
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        
        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)
        
        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)
        
        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)
        
        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape[0], 1, sk_out.shape[1])
        
        return visual * vis_out, skeleton * sk_out

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