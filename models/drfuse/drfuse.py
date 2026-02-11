import math

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models import resnet50, ResNet50_Weights
from models.registry import ModelRegistry

from .ehr_transformer_drfuse import EHRTransformer  
from .ehr_lstm_drfuse import EHRLSTMEncoder  
from ..base import BaseFuseTrainer
from ..base.base_encoder import create_cxr_encoder
from torch.optim.lr_scheduler import ReduceLROnPlateau

@ModelRegistry.register('drfuse')
class DrFuse(BaseFuseTrainer):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        if not hasattr(self.hparams, 'task'):
            if getattr(self.hparams, 'num_classes', 1) == 25:
                self.hparams.task = 'phenotype'
            elif getattr(self.hparams, 'num_classes', 1) == 7:
                self.hparams.task = 'los'
            else:
                self.hparams.task = 'mortality'
        
        if self.hparams.task == 'los':
            self.num_classes = 7
        elif self.hparams.task == 'phenotype':
            self.num_classes = self.hparams.num_classes
        else:
            self.num_classes = 1

        ehr_encoder_type = getattr(self.hparams, 'ehr_encoder', 'transformer')
        if ehr_encoder_type.lower() == 'lstm':
            self.ehr_model = EHRLSTMEncoder(
                input_size=self.hparams.input_dim, 
                num_classes=self.num_classes,
                hidden_size=self.hparams.hidden_size,
                num_layers_feat=self.hparams.ehr_n_layers_feat,
                num_layers_shared=self.hparams.ehr_n_layers_shared,
                num_layers_distinct=self.hparams.ehr_n_layers_distinct,
                dropout=self.hparams.ehr_dropout,
                bidirectional=getattr(self.hparams, 'ehr_lstm_bidirectional', True)
            )
        elif ehr_encoder_type.lower() == 'transformer':
            self.ehr_model = EHRTransformer(
                input_size=self.hparams.input_dim, 
                num_classes=self.num_classes,
                d_model=self.hparams.hidden_size, 
                n_head=self.hparams.ehr_n_head,
                n_layers_feat=self.hparams.ehr_n_layers_feat, 
                n_layers_shared=self.hparams.ehr_n_layers_shared,
                n_layers_distinct=self.hparams.ehr_n_layers_distinct,
                dropout=self.hparams.ehr_dropout
            )
        else:
            raise ValueError(f"Unsupported EHR encoder type: {ehr_encoder_type}. Supported types: 'lstm', 'transformer'")

        cxr_encoder_type = getattr(self.hparams, 'cxr_encoder', 'resnet50')
        pretrained = getattr(self.hparams, 'pretrained', True)
        
        # Use factory function to create CXR encoders for better code reuse
        # Create two independent instances: one for shared features, one for distinct features
        self.cxr_model_shared = create_cxr_encoder(
            encoder_type=cxr_encoder_type,
            hidden_size=self.hparams.hidden_size,
            pretrained=pretrained
        )
        
        self.cxr_model_spec = create_cxr_encoder(
            encoder_type=cxr_encoder_type,
            hidden_size=self.hparams.hidden_size,
            pretrained=pretrained
        )

        self.shared_project = nn.Sequential(
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size*2),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size*2, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)
        )

        self.ehr_model_linear = nn.Linear(in_features=self.hparams.hidden_size, out_features=self.num_classes)
        self.cxr_model_linear = nn.Linear(in_features=self.hparams.hidden_size, out_features=self.num_classes)
        self.fuse_model_shared = nn.Linear(in_features=self.hparams.hidden_size, out_features=self.num_classes)

        self.attn_proj = nn.Linear(self.hparams.hidden_size, (2+self.num_classes)*self.hparams.hidden_size)
        self.final_pred_fc = nn.Linear(self.hparams.hidden_size, self.num_classes)

        if self.hparams.task == 'los':
            self.pred_criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.pred_criterion = nn.BCELoss(reduction='none')
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.triplet_loss = nn.TripletMarginLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.jsd = JSD()

    def _compute_masked_pred_loss(self, input, target, mask):
        if self.hparams.task == 'los':
            target_1d = target.long().squeeze()
            return (self.pred_criterion(input, target_1d) * mask).sum() / max(mask.sum(), 1e-6)
        else:
            return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_abs_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y).abs() * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_mse(self, x, y, mask):
        return (self.mse_loss(x, y).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _disentangle_loss_mse(self, model_output, pairs, log=True, mode='train'):
        ehr_mask = torch.ones_like(pairs)
        loss_shared_alignment = self._masked_mse(model_output['feat_ehr_shared'],
                                                 model_output['feat_cxr_shared'], pairs)
        loss_sim_cxr = self._masked_abs_cos_sim(model_output['feat_cxr_shared'],
                                                model_output['feat_cxr_distinct'], pairs)
        loss_sim_ehr = self._masked_abs_cos_sim(model_output['feat_ehr_shared'],
                                                model_output['feat_ehr_distinct'], ehr_mask)

        loss_disentanglement = (self.hparams.lambda_disentangle_shared * loss_shared_alignment +
                                self.hparams.lambda_disentangle_ehr * loss_sim_ehr +
                                self.hparams.lambda_disentangle_cxr * loss_sim_cxr)
        if log:
            self.log_dict({
                f'disentangle_{mode}/EHR_disinct': loss_sim_ehr.detach(),
                f'disentangle_{mode}/CXR_disinct': loss_sim_cxr.detach(),
                f'disentangle_{mode}/shared_MSE': loss_shared_alignment.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=pairs.shape[0])

        return loss_disentanglement

    def _disentangle_loss_jsd(self, model_output, pairs, log=True, mode='train'):
        pairs = pairs.squeeze(1)
        ehr_mask = torch.ones_like(pairs)
        loss_sim_cxr = self._masked_abs_cos_sim(model_output['feat_cxr_shared'],
                                                model_output['feat_cxr_distinct'], pairs)
        loss_sim_ehr = self._masked_abs_cos_sim(model_output['feat_ehr_shared'],
                                                model_output['feat_ehr_distinct'], ehr_mask)

        jsd = self.jsd(model_output['feat_ehr_shared'].sigmoid(),
                       model_output['feat_cxr_shared'].sigmoid(), pairs)

        loss_disentanglement = (self.hparams.lambda_disentangle_shared * jsd +
                                self.hparams.lambda_disentangle_ehr * loss_sim_ehr +
                                self.hparams.lambda_disentangle_cxr * loss_sim_cxr)
        if log:
            self.log_dict({
                f'disentangle_{mode}/EHR_disinct': loss_sim_ehr.detach(),
                f'disentangle_{mode}/CXR_disinct': loss_sim_cxr.detach(),
                f'disentangle_{mode}/shared_jsd': jsd.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=pairs.shape[0])

        return loss_disentanglement

    def _disentangle_loss_adc(self, model_output, pairs, log=True, mode='train'):
        ehr_mask = torch.ones_like(pairs)
        domain_mask = torch.cat([pairs, pairs], dim=0)
        loss_adc = self._compute_masked_pred_loss(model_output['pred_domain'],
                                                  model_output['label_domain'], domain_mask)
        loss_align_mse = self._masked_mse(model_output['feat_ehr_shared'],
                                              model_output['feat_cxr_shared'], pairs)
        loss_shared_alignment = loss_adc

        loss_sim_cxr = self._masked_abs_cos_sim(model_output['feat_cxr_shared'],
                                                model_output['feat_cxr_distinct'], pairs)
        loss_sim_ehr = self._masked_abs_cos_sim(model_output['feat_ehr_shared'],
                                                model_output['feat_ehr_distinct'], ehr_mask)

        loss_disentanglement = (self.hparams.lambda_disentangle_shared * loss_shared_alignment +
                                self.hparams.lambda_disentangle_ehr * loss_sim_ehr +
                                self.hparams.lambda_disentangle_cxr * loss_sim_cxr)
        if log:
            self.log_dict({
                f'disentangle_{mode}/EHR_disinct': loss_sim_ehr.detach(),
                f'disentangle_{mode}/CXR_disinct': loss_sim_cxr.detach(),
                f'disentangle_{mode}/shared_domain_pred_loss': loss_adc.detach(),
                f'disentangle_{mode}/shared_MSE': loss_align_mse.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=pairs.shape[0])
        return loss_disentanglement

    def _disentangle_loss_triplet(self, model_output, pairs, log=True, mode='train'):
        
        triplet_loss_cxr = self.triplet_loss(model_output['feat_cxr_shared'],
                                             model_output['feat_ehr_shared'],
                                             model_output['feat_cxr_distinct'])
        triplet_loss_ehr = self.triplet_loss(model_output['feat_ehr_shared'],
                                             model_output['feat_cxr_shared'],
                                             model_output['feat_ehr_distinct'])
        triplet_loss_partial = self.triplet_loss(model_output['feat_ehr_shared'],
                                                 model_output['feat_ehr_shared'],
                                                 model_output['feat_ehr_distinct'])

        triplet_loss = pairs * (triplet_loss_cxr + triplet_loss_ehr) / 2 + (1 - pairs) * triplet_loss_partial

        loss_disentanglement = triplet_loss.mean()
        if log:
            self.log_dict({
                f'disentangle_{mode}/triplet_loss': loss_disentanglement.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=pairs.shape[0])

        return loss_disentanglement

    def _compute_prediction_losses(self, model_output, y_gt, pairs, log=True, mode='train'):
        ehr_mask = torch.ones_like(model_output['predictions'][:, 0])
        loss_pred_final = self._compute_masked_pred_loss(model_output['predictions'], y_gt, ehr_mask)
        loss_pred_ehr = self._compute_masked_pred_loss(model_output['pred_ehr'], y_gt, ehr_mask)
        pairs = pairs.squeeze(1)
        loss_pred_cxr = self._compute_masked_pred_loss(model_output['pred_cxr'], y_gt, pairs)
        loss_pred_shared = self._compute_masked_pred_loss(model_output['pred_shared'], y_gt, ehr_mask)

        if log:
            self.log_dict({
                f'{mode}_loss/pred_final': loss_pred_final.detach(),
                f'{mode}_loss/pred_shared': loss_pred_shared.detach(),
                f'{mode}_loss/pred_ehr': loss_pred_ehr.detach(),
                f'{mode}_loss/pred_cxr': loss_pred_cxr.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared

    def _compute_and_log_loss(self, model_output, y_gt, pairs, log=True, mode='train'):
 
        prediction_losses = self._compute_prediction_losses(model_output, y_gt, pairs, log, mode)
        loss_pred_final, loss_pred_ehr, loss_pred_cxr, loss_pred_shared = prediction_losses

        loss_prediction = (self.hparams.lambda_pred_shared * loss_pred_shared +
                           self.hparams.lambda_pred_ehr * loss_pred_ehr +
                           self.hparams.lambda_pred_cxr * loss_pred_cxr)

        if self.hparams.attn_fusion in ['mid', 'late']:
            loss_prediction = loss_pred_final + loss_prediction

        if self.hparams.disentangle_loss == 'mse':
            loss_disentanglement = self._disentangle_loss_mse(model_output, pairs, log, mode)
        elif self.hparams.disentangle_loss == 'adc':
            loss_disentanglement = self._disentangle_loss_adc(model_output, pairs, log, mode)
        elif self.hparams.disentangle_loss == 'triplet':
            loss_disentanglement = self._disentangle_loss_triplet(model_output, pairs, log, mode)
        elif self.hparams.disentangle_loss == 'jsd':
            loss_disentanglement = self._disentangle_loss_jsd(model_output, pairs, log, mode)

        loss_total = loss_prediction + loss_disentanglement

        epoch_log = {}
        if self.hparams.attn_fusion == 'mid':
            if self.hparams.task == 'los':
                raw_pred_loss_ehr = F.cross_entropy(model_output['pred_ehr'].data, y_gt.long().squeeze(), reduction='none')
                raw_pred_loss_cxr = F.cross_entropy(model_output['pred_cxr'].data, y_gt.long().squeeze(), reduction='none')
                raw_pred_loss_shared = F.cross_entropy(model_output['pred_shared'].data, y_gt.long().squeeze(), reduction='none')
            else:
                raw_pred_loss_ehr = F.binary_cross_entropy(model_output['pred_ehr'].data, y_gt, reduction='none')
                raw_pred_loss_cxr = F.binary_cross_entropy(model_output['pred_cxr'].data, y_gt, reduction='none')
                raw_pred_loss_shared = F.binary_cross_entropy(model_output['pred_shared'].data, y_gt, reduction='none')

            pairs = pairs.unsqueeze(1) if pairs.dim() == 1 else pairs
            attn_weights = model_output['attn_weights']
            
            if self.hparams.task == 'los':
                attn_ehr = attn_weights[:, :, 0].mean(dim=1)
                attn_shared = attn_weights[:, :, 1].mean(dim=1)
                attn_cxr = attn_weights[:, :, 2].mean(dim=1)
            elif self.hparams.task == 'phenotype':
                attn_ehr = attn_weights[:, :, 0].mean(dim=1)
                attn_shared = attn_weights[:, :, 1].mean(dim=1)
                attn_cxr = attn_weights[:, :, 2].mean(dim=1)
                raw_pred_loss_ehr = raw_pred_loss_ehr.mean(dim=1)
                raw_pred_loss_cxr = raw_pred_loss_cxr.mean(dim=1)
                raw_pred_loss_shared = raw_pred_loss_shared.mean(dim=1)
            else:
                attn_ehr, attn_shared, attn_cxr = attn_weights[:, :, 0], attn_weights[:, :, 1], attn_weights[:, :, 2]

            cxr_overweights_ehr = 2 * (raw_pred_loss_cxr < raw_pred_loss_ehr).float() - 1
            loss_attn1 = pairs.squeeze() * F.margin_ranking_loss(attn_cxr, attn_ehr, cxr_overweights_ehr, reduction='none')
            loss_attn1 = loss_attn1.sum() / max(1e-6, loss_attn1[loss_attn1>0].numel())

            shared_overweights_ehr = 2 * (raw_pred_loss_shared < raw_pred_loss_ehr).float() - 1
            loss_attn2 = pairs.squeeze() * F.margin_ranking_loss(attn_shared, attn_ehr, shared_overweights_ehr, reduction='none')
            loss_attn2 = loss_attn2.sum() / max(1e-6, loss_attn2[loss_attn2>0].numel())

            shared_overweights_cxr = 2 * (raw_pred_loss_shared < raw_pred_loss_cxr).float() - 1
            loss_attn3 = pairs.squeeze() * F.margin_ranking_loss(attn_shared, attn_cxr, shared_overweights_cxr, reduction='none')
            loss_attn3 = loss_attn3.sum() / max(1e-6, loss_attn3[loss_attn3>0].numel())

            loss_attn_ranking = (loss_attn1 + loss_attn2 + loss_attn3) / 3

            loss_total = loss_total + self.hparams.lambda_attn_aux * loss_attn_ranking
            epoch_log[f'{mode}_loss/attn_aux'] = loss_attn_ranking.detach()

        if log:
            epoch_log.update({
                f'{mode}_loss/total': loss_total.detach(),
                f'{mode}_loss/prediction': loss_prediction.detach(),
                'step': float(self.current_epoch)
            })
            self.log_dict(epoch_log, on_epoch=True, on_step=False, batch_size=y_gt.shape[0], sync_dist=True)

        return loss_total


    def forward(self, data_dict):
        x = data_dict['ehr_ts']
        img = data_dict['cxr_imgs']
        seq_lengths = data_dict['seq_len']
        pairs = data_dict['has_cxr']

        feat_ehr_shared, feat_ehr_distinct, pred_ehr = self.ehr_model(x, seq_lengths)
        feat_cxr_shared = self.cxr_model_shared(img)
        feat_cxr_distinct = self.cxr_model_spec(img)

        if self.hparams.task == 'los':
            pred_cxr = self.cxr_model_linear(feat_cxr_distinct)
        else:
            pred_cxr = self.cxr_model_linear(feat_cxr_distinct).sigmoid()

        feat_ehr_shared = self.shared_project(feat_ehr_shared)
        feat_cxr_shared = self.shared_project(feat_cxr_shared)

        pairs = pairs.unsqueeze(1)
        
        if self.hparams.logit_average:
            h1 = feat_ehr_shared
            h2 = feat_cxr_shared
            term1 = torch.stack([h1+h2, h1+h2, h1, h2], dim=2)
            term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
            feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)
        else:
            feat_avg_shared = (feat_ehr_shared + feat_cxr_shared) / 2
        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * feat_ehr_shared
        if self.hparams.task == 'los':
            pred_shared = self.fuse_model_shared(feat_avg_shared)
        else:
            pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()

        attn_input = torch.stack([feat_ehr_distinct, feat_avg_shared, feat_cxr_distinct], dim=1)
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2+self.num_classes, dim=-1)

        q_mean = pairs * q.mean(dim=1) + (1-pairs) * q[:, :-1].mean(dim=1)

        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])

        attn_mask = torch.ones_like(attn_logits)
        attn_mask[(pairs.squeeze(-1)==0).nonzero(as_tuple=False).squeeze(), :, -1] = 0
        attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)

        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.final_pred_fc(feat_final)
        if self.hparams.task == 'los':
            pred_cxr = self.cxr_model_linear(feat_cxr_distinct)
            pred_shared = self.fuse_model_shared(feat_avg_shared)
            pred_final = torch.diagonal(pred_final, dim1=1, dim2=2)
        else:
            pred_cxr = self.cxr_model_linear(feat_cxr_distinct).sigmoid()
            pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()
            pred_final = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()

        outputs = {
            'feat_ehr_shared': feat_ehr_shared,
            'feat_cxr_shared': feat_cxr_shared,
            'feat_ehr_distinct': feat_ehr_distinct,
            'feat_cxr_distinct': feat_cxr_distinct,
            'feat_final': feat_final,
            'predictions': pred_final,
            'pred_shared': pred_shared,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
            'attn_weights': attn_weights,
        }

        outputs['loss'] = self._compute_and_log_loss(outputs, data_dict['labels'], pairs)

        return outputs
    
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


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor, masks):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum() / max(1e-6, masks.sum())