import torch
from torch import nn

from models.registry import ModelRegistry

from ..base import BaseFuseTrainer
from .cross_vpt import CXRDynamicPromptViT, EHRPromptGenerator


@ModelRegistry.register('crossvpt')
class CrossVPTLightning(BaseFuseTrainer):
    """CareBench wrapper for CrossVPT.

    Notes:
    - Keeps CrossVPT architecture intact (EHR encoder -> prompts -> ViT injection -> heads).
    - Adapts loss/activation per task to match CareBench tasks:
      - mortality/phenotype: sigmoid + BCE
      - los: softmax + CrossEntropy
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Auto-detect task from num_classes if not provided
        if not hasattr(self.hparams, 'task'):
            if getattr(self.hparams, 'num_classes', 1) == 25:
                self.hparams.task = 'phenotype'
            elif getattr(self.hparams, 'num_classes', 1) == 7:
                self.hparams.task = 'los'
            elif getattr(self.hparams, 'num_classes', 1) == 9:
                self.hparams.task = 'phenotype'  # Phenotype-9
            else:
                self.hparams.task = 'mortality'

        self.task = self.hparams.task
        self.class_names = getattr(self.hparams, 'class_names', None)
        
        # Auto-detect num_classes based on task and actual label size
        if self.task == 'los':
            self.num_classes = 7
        elif self.task == 'phenotype':
            # Support both Phenotype-48 (25 classes) and Phenotype-9 (9 classes)
            self.num_classes = getattr(self.hparams, 'num_classes', 25)
        else:
            self.num_classes = 1

        self.cxr_vit = CXRDynamicPromptViT(
            vit_feature=getattr(self.hparams, 'vpt_feature', 'sup_vitb16_imagenet21k'),
            vit_model_root=getattr(self.hparams, 'vpt_model_root', ''),
            crop_size=getattr(self.hparams, 'vpt_crop_size', 224),
            load_pretrain=getattr(self.hparams, 'load_pretrain', True),
            freeze_vit=getattr(self.hparams, 'freeze_vit', True),
            bias_tune=getattr(self.hparams, 'bias_tune', False),
            partial_layers=getattr(self.hparams, 'partial_layers', 0),
            hf_model_id=getattr(self.hparams, 'hf_model_id', None),
        )
        vit_hidden_dim = self.cxr_vit.vit_hidden_dim

        self.w_multi = float(getattr(self.hparams, 'loss_multi', 1.0))
        self.w_ehr = float(getattr(self.hparams, 'aux_ehr_weight', 0.0))
        self.w_cxr = float(getattr(self.hparams, 'loss_cxr', 1.0))

        ehr_prompt_hparams = {
            'prompt_noise_std': getattr(self.hparams, 'prompt_noise_std', 0.0),
            'prompt_token_dropout': getattr(self.hparams, 'prompt_token_dropout', 0.0),
        }
        ehr_input_size = getattr(self.hparams, 'ehr_input_size', None)
        if ehr_input_size is None:
            ehr_input_size = getattr(self.hparams, 'input_dim', 24)

        self.ehr_prompt_gen = EHRPromptGenerator(
            ehr_input_size=ehr_input_size,
            ehr_hidden_size=getattr(self.hparams, 'hidden_size', 256),
            num_classes=self.num_classes,
            num_prompt_tokens=getattr(self.hparams, 'num_prompt_tokens', 5),
            vit_hidden_dim=vit_hidden_dim,
            ehr_n_head=getattr(self.hparams, 'ehr_n_head', 4),
            ehr_n_layers_distinct=getattr(self.hparams, 'ehr_n_layers_distinct', 1),
            ehr_dropout=getattr(self.hparams, 'ehr_dropout', 0.2),
            prompt_project_dim=getattr(self.hparams, 'prompt_project_dim', -1),
            prompt_dropout=getattr(self.hparams, 'prompt_dropout', 0.0),
            simple=True,
            w_ehr=self.w_ehr,
            hparams=ehr_prompt_hparams,
        )

        self.cxr_head = nn.Linear(vit_hidden_dim, self.num_classes)
        self.fusion_head = nn.Linear(vit_hidden_dim + getattr(self.hparams, 'hidden_size', 256), self.num_classes)

        self.fusion_variant = getattr(self.hparams, 'fusion_variant', None)

        if hasattr(self.hparams, 'pool_include'):
            self.pool_include = set(getattr(self.hparams, 'pool_include', ['cls', 'prompt', 'image']))
        else:
            pool_exclude = set(getattr(self.hparams, 'pool_exclude', ['cls', 'prompt']))
            all_options = {'cls', 'prompt', 'image'}
            if 'patches' in pool_exclude:
                pool_exclude.add('image')
                pool_exclude.discard('patches')
            self.pool_include = all_options - pool_exclude
        self.num_prompt_tokens = getattr(self.hparams, 'num_prompt_tokens', 5)

        if self.fusion_variant == 'concat_avgpool':
            in_dim = vit_hidden_dim + getattr(self.hparams, 'hidden_size', 256)
            self.fusion_head_avg = nn.Linear(in_dim, self.num_classes)

        if self.fusion_variant == 'cls_residual':
            self.ehr2vit = nn.Linear(getattr(self.hparams, 'hidden_size', 256), vit_hidden_dim)
            alpha_init = float(getattr(self.hparams, 'alpha_init', 0.1))
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        if self.task == 'los':
            self.pred_criterion = nn.CrossEntropyLoss()
        else:
            self.pred_criterion = nn.BCELoss()

    def _postprocess_logits(self, logits: torch.Tensor):
        if self.task == 'los':
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.task == 'los':
            labels = labels.view(-1).long()
            return self.pred_criterion(logits, labels)
        return self.pred_criterion(torch.sigmoid(logits), labels)

    def forward(self, data_dict):
        ehr_ts = data_dict['ehr_ts']
        seq_len = data_dict['seq_len']
        cxr_imgs = data_dict['cxr_imgs']
        labels = data_dict['labels']

        ehr_feat, ehr_prompts, pred_ehr = self.ehr_prompt_gen(ehr_ts, seq_len)

        cxr_feat, encoded = self.cxr_vit(cxr_imgs, ehr_prompts)
        cxr_logits = self.cxr_head(cxr_feat)
        pred_cxr = self._postprocess_logits(cxr_logits)

        if self.task == 'los':
            # EHR auxiliary pred: use the same linear head, but softmax
            if getattr(self.ehr_prompt_gen, 'ehr_head', None) is not None:
                ehr_logits = self.ehr_prompt_gen.ehr_head(ehr_feat)
                pred_ehr = torch.softmax(ehr_logits, dim=-1)
            else:
                pred_ehr = None

        variant = self.fusion_variant
        if variant == 'concat_avgpool':
            tokens_list = []
            if 'cls' in self.pool_include:
                tokens_list.append(encoded[:, 0:1, :])
            if 'prompt' in self.pool_include:
                prompt_start = 1
                prompt_end = 1 + self.num_prompt_tokens
                tokens_list.append(encoded[:, prompt_start:prompt_end, :])
            if 'image' in self.pool_include or 'patches' in self.pool_include:
                image_start = 1 + self.num_prompt_tokens
                tokens_list.append(encoded[:, image_start:, :])
            if len(tokens_list) == 0:
                raise ValueError("pool_include 不能为空，至少需要包含 'cls', 'prompt', 或 'image'/'patches' 之一")
            img_tokens = tokens_list[0] if len(tokens_list) == 1 else torch.cat(tokens_list, dim=1)
            img_avg = img_tokens.mean(dim=1)
            concat = torch.cat([img_avg, ehr_feat], dim=1)

            fusion_logits = self.fusion_head_avg(concat)
            pred_multi = self._postprocess_logits(fusion_logits)

            w_multi = float(getattr(self.hparams, 'loss_multi', 1.0))
            w_ehr = float(getattr(self.hparams, 'loss_ehr', 1.0))
            w_cxr = float(getattr(self.hparams, 'loss_cxr', 1.0))

            loss_multi = self._compute_loss(fusion_logits, labels)
            loss_ehr = 0.0
            if pred_ehr is not None:
                if self.task == 'los':
                    loss_ehr = self.pred_criterion(self.ehr_prompt_gen.ehr_head(ehr_feat), labels.view(-1).long())
                else:
                    loss_ehr = self.pred_criterion(pred_ehr, labels)
            loss_cxr = self._compute_loss(cxr_logits, labels)
            loss = w_multi * loss_multi + w_ehr * loss_ehr + w_cxr * loss_cxr

        elif variant == 'cls_residual':
            ehr_proj = self.ehr2vit(ehr_feat)
            fusion_cls = cxr_feat + self.alpha * ehr_proj

            fusion_logits = self.cxr_head(fusion_cls)
            pred_multi = self._postprocess_logits(fusion_logits)

            w_multi = float(getattr(self.hparams, 'loss_multi', 1.0))
            w_ehr = float(getattr(self.hparams, 'aux_ehr_weight', 0.0))
            w_cxr = float(getattr(self.hparams, 'loss_cxr', 0.0))

            loss_multi = self._compute_loss(fusion_logits, labels)
            loss_ehr = 0.0
            if pred_ehr is not None:
                if self.task == 'los':
                    loss_ehr = self.pred_criterion(self.ehr_prompt_gen.ehr_head(ehr_feat), labels.view(-1).long())
                else:
                    loss_ehr = self.pred_criterion(pred_ehr, labels)
            loss_cxr = self._compute_loss(cxr_logits, labels)
            loss = w_multi * loss_multi + w_ehr * loss_ehr + w_cxr * loss_cxr

        else:
            final_mode = getattr(self.hparams, 'final_pred_mode', 'concat')
            if final_mode == 'cxr_only':
                fusion_logits = cxr_logits
                pred_multi = pred_cxr

                w_multi = float(getattr(self.hparams, 'loss_multi', 1.0))
                w_ehr = float(getattr(self.hparams, 'aux_ehr_weight', 0.0))

                loss_multi = self._compute_loss(fusion_logits, labels)
                loss_ehr = 0.0
                if pred_ehr is not None:
                    if self.task == 'los':
                        loss_ehr = self.pred_criterion(self.ehr_prompt_gen.ehr_head(ehr_feat), labels.view(-1).long())
                    else:
                        loss_ehr = self.pred_criterion(pred_ehr, labels)
                loss = w_multi * loss_multi + w_ehr * loss_ehr
            else:
                fused = torch.cat([ehr_feat, cxr_feat], dim=1)

                fusion_logits = self.fusion_head(fused)
                pred_multi = self._postprocess_logits(fusion_logits)

                w_multi = float(getattr(self.hparams, 'loss_multi', 1.0))
                w_ehr = float(getattr(self.hparams, 'loss_ehr', 1.0))
                w_cxr = float(getattr(self.hparams, 'loss_cxr', 1.0))

                loss_multi = self._compute_loss(fusion_logits, labels)
                loss_ehr = 0.0
                if pred_ehr is not None:
                    if self.task == 'los':
                        loss_ehr = self.pred_criterion(self.ehr_prompt_gen.ehr_head(ehr_feat), labels.view(-1).long())
                    else:
                        loss_ehr = self.pred_criterion(pred_ehr, labels)
 
                loss_cxr = self._compute_loss(cxr_logits, labels)
                loss = w_multi * loss_multi + w_ehr * loss_ehr + w_cxr * loss_cxr

        return {
            'predictions': pred_multi,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
            'feat_ehr_distinct': ehr_feat,
            'feat_cxr_distinct': cxr_feat,
            'loss': loss,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(getattr(self.hparams, 'lr', 0.0001)),
            weight_decay=float(getattr(self.hparams, 'wd', 0.0)),
        )
        return optimizer

    def _val_test_shared_step(self, batch, cache):
        """
        重写基类方法以支持 CrossVPT 的多分支评估（pred_ehr, pred_cxr）。
        其他逻辑（meta_attrs, groups）使用基类实现。
        """
        out = self._shared_step(batch)
        
        # 主预测（所有模型都需要）
        cache['predictions'].append(out['predictions'].detach())
        cache['labels'].append(batch['labels'].detach())
        
        # CrossVPT 特有的多分支预测（用于评估各模态的贡献）
        # 注意：pred_ehr 可能为 None（当 aux_ehr_weight <= 0.0 时）
        if out.get('pred_ehr') is not None:
            cache['pred_ehr'].append(out['pred_ehr'].detach())
        if out.get('pred_cxr') is not None:
            cache['pred_cxr'].append(out['pred_cxr'].detach())
        
        # 使用基类逻辑处理 meta_attrs 和 groups（用于公平性指标）
        if 'meta_attrs' in batch:
            if 'meta_attrs' not in cache:
                cache['meta_attrs'] = []
            meta_attrs_list = batch['meta_attrs'].to_dict('records')
            cache['meta_attrs'].extend(meta_attrs_list)
        if 'groups' in batch:
            cache['groups'].extend(batch['groups'])
        
        return out

