import math
import copy

import torch
from torch import nn

import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Parameter

from ..base.base_encoder import create_ehr_encoder, create_cxr_encoder

from models.registry import ModelRegistry
from ..base import BaseFuseTrainer


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features).float())
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x):
        y = torch.mm(x.float(), self.weight.float())
        output = torch.mm(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float()
        else:
            return output


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def euclidean_dist(x, y):
    b = x.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(b, b)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(b, b).t()
    dist = xx+yy-2*torch.mm(x, y.t())
    return dist

def guassian_kernel(source, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
    n = source.size(0)
    L2_distance = euclidean_dist(source, source)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n**2-n)
    
    if bandwidth < 1e-3:
        bandwidth = 1
    
    bandwidth /= kernel_mul ** (kernel_num//2)
    bandwidth_list = [bandwidth*(kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance/bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)/len(kernel_val)


@ModelRegistry.register('m3care') 
class M3Care(BaseFuseTrainer):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        if self.hparams.task == 'phenotype':
            self.num_classes = self.hparams.num_classes
        elif self.hparams.task == 'mortality':
            self.num_classes = 1
        elif self.hparams.task == 'los':
            self.num_classes = 7  # LoS has 7 classes (bins 2-8, excluding 0,1)
        else:
            raise ValueError(f"Unsupported task: {self.hparams.task}. Only 'mortality', 'phenotype', and 'los' are supported")

        self.input_dim = self.hparams.input_dim 
        self.hidden_dim = self.hparams.hidden_dim
        self.dropout = self.hparams.dropout
        self.lmbda = self.hparams.stab_reg_lambda
        self.modal_num = 2

        # Set encoder defaults if not specified
        if not hasattr(self.hparams, 'ehr_encoder'):
            self.hparams.ehr_encoder = 'transformer'  # M3Care originally uses transformer
        if not hasattr(self.hparams, 'cxr_encoder'):
            self.hparams.cxr_encoder = 'resnet50'
        if not hasattr(self.hparams, 'pretrained'):
            self.hparams.pretrained = True

        self._init_encoders()
        self._init_m3care_components()

    def _init_encoders(self):
        """Initialize configurable encoders"""
        # Get encoder types
        ehr_encoder_type = getattr(self.hparams, 'ehr_encoder', 'transformer').lower()
        cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50').lower()
        pretrained = getattr(self.hparams, 'pretrained', True)

        # ===== EHR Encoder Selection using Factory =====
        ehr_params = {
            'input_size': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_dim,
            'dropout': getattr(self.hparams, 'ehr_dropout', self.dropout),
        }
        
        # Add encoder-specific parameters
        if ehr_encoder_type == 'lstm':
            ehr_params.update({
                'num_layers': getattr(self.hparams, 'ehr_num_layers', 2),
                'bidirectional': getattr(self.hparams, 'ehr_bidirectional', True)
            })
        elif ehr_encoder_type == 'transformer':
            ehr_params.update({
                'd_model': ehr_params.pop('hidden_size'),
                'n_head': getattr(self.hparams, 'ehr_n_head', 8),
                'n_layers': getattr(self.hparams, 'ehr_n_layers', 2),
                'max_len': getattr(self.hparams, 'max_len', 350)
            })
        
        # Create EHR encoder
        self.ehr_encoder = create_ehr_encoder(encoder_type=ehr_encoder_type, **ehr_params)

        # ===== CXR Encoder Selection using Factory =====
        cxr_params = {
            'hidden_size': self.hidden_dim,
            'pretrained': pretrained
        }
        
        # Create CXR encoder
        self.cxr_encoder = create_cxr_encoder(encoder_type=cxr_encoder_type, **cxr_params)

        print(f"M3Care model initialized:")
        print(f"  - EHR encoder: {ehr_encoder_type}, hidden_dim: {self.hidden_dim}")
        print(f"  - CXR encoder: {cxr_encoder_type}, hidden_dim: {self.hidden_dim}")
        print(f"  - Task: {self.hparams.task}, Pretrained: {pretrained}")

    def _init_m3care_components(self):
        """Initialize M3Care specific components"""
        self.dropout = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.proj1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, self.num_classes)
        
        self.threshold = nn.Parameter(torch.ones(size=(1,)) + 1)        
        self.bn = nn.BatchNorm1d(self.hidden_dim)
    
        self.simiProj = clones(torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            self.relu,
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            self.relu,
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
        ), self.modal_num)
        
        self.GCN = clones(GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True), self.modal_num)
        self.GCN_2 = clones(GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True), self.modal_num)
        self.GCN_3 = clones(GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True), self.modal_num)
        
        self.eps0 = nn.Parameter(torch.ones(size=(1,)) + 1)
        self.eps1 = nn.Parameter(torch.ones(size=(1,)) + 1)
        self.eps2 = nn.Parameter(torch.ones(size=(1,)) + 1)
        
        self.weight1 = clones(nn.Linear(self.hidden_dim, 1), self.modal_num)
        self.weight2 = clones(nn.Linear(self.hidden_dim, 1), self.modal_num)

    def forward(self, data_dict):
        ehr = data_dict['ehr_ts']
        cxr = data_dict['cxr_imgs']
        seq_lengths = data_dict['seq_len']
        pairs = data_dict['has_cxr']  # [batch,]

        cxr_mask = pairs.unsqueeze(0)
        ehr_sim_mask = torch.ones_like(cxr_mask)
        ehr_sim_mask = ehr_sim_mask * ehr_sim_mask.permute(1, 0)
        cxr_mask = cxr_mask * cxr_mask.permute(1, 0)

        # Use configurable encoders
        ehr_feat_avg, _ = self.ehr_encoder(ehr, seq_lengths, output_prob=False)
        cxr_feat = self.cxr_encoder(cxr)

        ehr_hidden_mat = guassian_kernel(self.bn(self.relu(self.simiProj[0](ehr_feat_avg))), kernel_mul=2.0, kernel_num=3)
        ehr_hidden_mat2 = guassian_kernel(self.bn(ehr_feat_avg), kernel_mul=2.0, kernel_num=3)
        ehr_hidden_mat = ((1-self.sigmoid(self.eps0))*ehr_hidden_mat+self.sigmoid(self.eps0))*ehr_hidden_mat2
        ehr_hidden_mat = ehr_hidden_mat*ehr_sim_mask

        cxr_hidden_mat = guassian_kernel(self.bn(self.relu(self.simiProj[1](cxr_feat))), kernel_mul=2.0, kernel_num=3)
        cxr_hidden_mat2 = guassian_kernel(self.bn(cxr_feat), kernel_mul=2.0, kernel_num=3)
        cxr_hidden_mat = ((1-self.sigmoid(self.eps1))*cxr_hidden_mat+self.sigmoid(self.eps1))*cxr_hidden_mat2
        cxr_hidden_mat = cxr_hidden_mat*cxr_mask

        diff1 = torch.abs(torch.norm(self.simiProj[0](ehr_feat_avg)) - torch.norm(ehr_feat_avg))
        diff2 = torch.abs(torch.norm(self.simiProj[1](cxr_feat)) - torch.norm(cxr_feat))

        sum_of_diff = diff1+diff2

        similar_score = (ehr_hidden_mat+cxr_hidden_mat)/ \
        (torch.ones_like(cxr_mask)+ torch.ones_like(cxr_mask) + torch.ones_like(cxr_mask) \
         + cxr_mask +  ehr_sim_mask )

        similar_score = self.relu(similar_score - self.sigmoid(self.threshold)[0])  
        temp_thresh = self.sigmoid(self.threshold)[0]
        bin_mask = similar_score>0
        similar_score = similar_score + bin_mask * temp_thresh.detach()

        ehr_hidden0 = self.relu(self.GCN[0](similar_score*ehr_sim_mask, ehr_feat_avg))
        ehr_hidden0 = self.relu(self.GCN_2[0](similar_score*ehr_sim_mask, ehr_hidden0))

        cxr_hidden0 = self.relu(self.GCN[1](similar_score*cxr_mask, cxr_feat))
        cxr_hidden0 = self.relu(self.GCN_2[1](similar_score*cxr_mask, cxr_hidden0))

        ehr_weight1=torch.sigmoid(self.weight1[0](ehr_hidden0))
        ehr_weight2 = torch.sigmoid(self.weight2[0](ehr_feat_avg ))
        ehr_weight1 = ehr_weight1/(ehr_weight1+ehr_weight2)
        ehr_weight2= 1-ehr_weight1

        cxr_weight1=torch.sigmoid(self.weight1[1](cxr_hidden0))
        cxr_weight2 = torch.sigmoid(self.weight2[1](cxr_feat ))
        cxr_weight1 = cxr_weight1/(cxr_weight1+cxr_weight2)
        cxr_weight2= 1-cxr_weight1

        final_ehr = ehr_weight1*ehr_hidden0+ehr_weight2*ehr_feat_avg
        final_cxr = cxr_weight1*cxr_hidden0+cxr_weight2*cxr_feat

        final_cxr = pairs.unsqueeze(1) * cxr_feat + (1-pairs.unsqueeze(1)) * final_cxr

        combined_hidden = torch.cat((final_ehr, final_cxr), dim=-1)

        last_hs_proj = self.dropout(F.relu(self.proj1(combined_hidden)))
        
        output = self.out_layer(last_hs_proj)

        y = data_dict['labels'].squeeze()
        pred_loss = self.classification_loss(output, y)
        total_loss = pred_loss + self.lmbda * sum_of_diff

        out_dict = {
            'loss': total_loss,
            'pred_loss': pred_loss,
            'predictions': output,
            'labels': y,
            'stability_regularization': sum_of_diff
        }

        return out_dict

    def training_step(self, batch, batch_idx):
        out = self(batch)

        self.log_dict({'train/prediction_loss': out['pred_loss'].detach()}, on_epoch=True, on_step=True,
                      batch_size=out['labels'].shape[0], sync_dist=True)
        self.log_dict({'train/stability_regularization': out['stability_regularization'].detach()}, on_epoch=True, on_step=True,
                      batch_size=out['labels'].shape[0], sync_dist=True)
        self.log_dict({'train/total_loss': out['loss'].detach()}, on_epoch=True, on_step=True,
                      batch_size=out['labels'].shape[0], sync_dist=True)
        
        return out
    
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