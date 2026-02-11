import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import ResNet
    
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

class DAFTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ndim_non_img: int = 15,
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 7,
    ) -> None:
        super(DAFTBlock, self).__init__()
        self.scale_activation = None
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None

        self.location = location
        self.film_dims = in_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
        self.bottleneck_dim = bottleneck_dim
        aux_input_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def forward(self, x_aux, feature_map):
        ehr_avg = torch.mean(x_aux, dim=1)
        
        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, ehr_avg), dim=1)

        attention = self.aux(squeeze)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(v_scale.size()[0], 1, v_scale.size()[1]).expand_as(x_aux)
            v_shift = v_shift.view(v_shift.size()[0], 1, v_shift.size()[1]).expand_as(x_aux)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(v_scale.size()[0], 1, v_scale.size()[1]).expand_as(x_aux)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(v_shift.size()[0], 1, v_shift.size()[1]).expand_as(x_aux)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * x_aux) + v_shift

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