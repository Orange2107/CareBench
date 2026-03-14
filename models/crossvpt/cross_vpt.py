import os
import sys
from typing import Tuple

import torch
from torch import nn

from .ehr_transformer import DisentangledEHRTransformer
# 使用 CareBench 完整版基类，获得完整的功能支持（特征保存、公平性指标、三任务支持等）
from ..base import BaseFuseTrainer


def _ensure_vpt_on_path():
    here = os.path.abspath(os.path.dirname(__file__))
    vpt_src = os.path.abspath(os.path.join(here, '..', 'vpt-main', 'src'))
    if vpt_src not in sys.path:
        sys.path.insert(0, vpt_src)


class EHRPromptGenerator(nn.Module):
    """Encodes EHR time series and produces (B, P, D_vit) prompt tokens.

    - Uses DisentangledEHRTransformer to get an EHR representation (B, H_ehr).
    - Projects to prompt tokens via MLP: H_ehr -> (P * D_vit), then reshape to (B, P, D_vit).
    """

    def __init__(
        self,
        ehr_input_size: int,
        ehr_hidden_size: int,
        num_classes: int,
        num_prompt_tokens: int,
        vit_hidden_dim: int,
        ehr_n_head: int = 4,
        ehr_n_layers_distinct: int = 1,
        ehr_dropout: float = 0.2,
        prompt_project_dim: int = -1,
        prompt_dropout: float = 0.0,
        simple: bool = True,
        w_ehr: float = 0.0,
        hparams: dict = None,
    ):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.vit_hidden_dim = vit_hidden_dim

        # EHR encoder
        self.ehr_model = DisentangledEHRTransformer(
            input_size=ehr_input_size,
            num_classes=num_classes,
            d_model=ehr_hidden_size,
            n_head=ehr_n_head,
            n_layers_feat=1,
            n_layers_shared=1,
            n_layers_distinct=ehr_n_layers_distinct,
            dropout=ehr_dropout,
            simple=simple,
        )

        # 三层渐进式 MLP（所有情况都使用）
        # 维度流动：256 → 768 → (ceil(P/2)×768) → (P×768)
        # 使用向上取整确保中间维度不会太小
        import math
        num_prompts = num_prompt_tokens
        hidden_dim_2 = math.ceil(num_prompts / 2) * 768  # 第 2 层输出：ceil(P/2) × 768
        
        self.prompt_proj = nn.Sequential(
            # 第 1 层：256 → 768
            nn.Linear(ehr_hidden_size, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            
            # 第 2 层：768 → ceil(P/2)×768（渐进扩大）
            nn.Linear(768, hidden_dim_2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim_2),
            nn.Dropout(0.1),
            
            # 第 3 层：ceil(P/2)×768 → P×768（最终输出）
            nn.Linear(hidden_dim_2, num_prompt_tokens * vit_hidden_dim),
        )

        # Prompt 归一化（在注入 ViT 前）
        self.prompt_norm = nn.LayerNorm(vit_hidden_dim)

        # 轻度残差机制
        self.base_prompts = nn.Parameter(torch.randn(num_prompt_tokens, vit_hidden_dim) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # ✨ 新增：Prompt 噪声和 Dropout 控制
        hparams = hparams or {}
        self.prompt_noise_std = hparams.get('prompt_noise_std', 0.0)
        self.prompt_token_dropout = hparams.get('prompt_token_dropout', 0.0)

        # Heads for per-modality prediction (for logging/comparison)
        # 
        self.ehr_head = nn.Linear(ehr_hidden_size, num_classes) if w_ehr > 0.0 else None

    def forward(self, ehr_ts: torch.Tensor, seq_len: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: ehr_feat, ehr_prompts, pred_ehr
        - ehr_feat: (B, H_ehr)
        - ehr_prompts: (B, P, D_vit)
        - pred_ehr: (B, C)
        """
        ehr_feat, pred_ehr = self.ehr_model(ehr_ts, seq_len) # 得到 EHR 特征 [batch, feature]
        # pred_ehr already sigmoid from DisentangledEHRTransformer

        # MLP 映射（三层渐进式）
        prompt_flat = self.prompt_proj(ehr_feat)  # (B, num_prompt_tokens * vit_hidden_dim)
        B = prompt_flat.shape[0]

        # Reshape 为 prompt tokens
        ehr_prompts = prompt_flat.view(B, self.num_prompt_tokens, self.vit_hidden_dim)

        # LayerNorm（关键！确保与 ViT patch tokens 尺度一致）
        ehr_prompts = self.prompt_norm(ehr_prompts)

        # 轻度残差（可选）
        ehr_prompts = self.base_prompts + self.alpha * ehr_prompts

        # ✨ 新增：训练时添加 Prompt 噪声（类似 GAVS）
        if self.training and self.prompt_noise_std > 0:
            noise = torch.randn_like(ehr_prompts) * self.prompt_noise_std
            ehr_prompts = ehr_prompts + noise
            # 可选：打印噪声统计信息（调试用）
            # print(f"Prompt noise: std={noise.std().item():.4f}")

        # ✨ 新增：Prompt Token Dropout（训练时随机丢弃部分 tokens）
        if self.training and self.prompt_token_dropout > 0:
            mask = torch.bernoulli(torch.ones(B, self.num_prompt_tokens, 1, device=ehr_prompts.device) * (1 - self.prompt_token_dropout))
            ehr_prompts = ehr_prompts * mask / (1 - self.prompt_token_dropout)  # 缩放保持期望值

        # Consistent EHR-head (may differ from ehr_model's built-in head)
        pred_ehr_head = torch.sigmoid(self.ehr_head(ehr_feat)) if self.ehr_head is not None else None

        return ehr_feat, ehr_prompts, pred_ehr_head


class CXRDynamicPromptViT(nn.Module):
    """Wraps a ViT backbone (from vpt-main or HuggingFace) and injects dynamic prompts from EHR.

    The ViT model itself is not modified. We call its embeddings and encoder
    directly to insert prompts: [CLS] + [prompts] + [patches].
    
    支持两种 ViT 后端：
    1. vpt-main ViT（原有逻辑）
    2. HuggingFace ViT（新增，支持 CheXpert 预训练模型）
    """

    def __init__(
        self,
        vit_feature: str, # VIT 模型类型
        vit_model_root: str, # VIT 模型权重路径
        crop_size: int, # 裁剪大小
        load_pretrain: bool = True, # 是否加载预训练权重
        freeze_vit: bool = True, # 是否冻结 VIT 模型
        bias_tune: bool = False, # 是否调整 bias
        partial_layers: int = 0, # 部分层微调
        hf_model_id: str = None, # 新增：HuggingFace 模型 ID
    ):
        super().__init__()
        
        # 判断是否使用 HuggingFace ViT
        if vit_feature == 'hf_chexpert_vit' or (vit_feature and vit_feature.startswith('hf_')):
            # 使用 HuggingFace ViT 包装器
            from .hf_vit_wrapper import CXRDynamicPromptViT_HF
            used_id = hf_model_id or "codewithdark/vit-chest-xray"
            self.vit = CXRDynamicPromptViT_HF(
                hf_model_id=used_id,
                freeze_vit=freeze_vit,
                bias_tune=bias_tune,
                partial_layers=partial_layers,
            )
            self.vit_hidden_dim = self.vit.vit_hidden_dim
            self.use_hf = True
            print(f"[CXRDynamicPromptViT] 使用 HuggingFace ViT: {used_id}")
        else:
            # 使用现有 vpt-main ViT
            _ensure_vpt_on_path()
            from .src.models.build_vit_backbone import build_vit_sup_models

            # Build ViT backbone (VisionTransformer)
            vit_model, feat_dim = build_vit_sup_models(
                model_type=vit_feature,
                crop_size=crop_size,
                prompt_cfg=None, #
                model_root=vit_model_root,
                adapter_cfg=None, # 是否使用 adapter
                load_pretrain=load_pretrain,
                vis=False,
            )
            self.vit = vit_model
            self.vit_hidden_dim = feat_dim
            self.use_hf = False

            # Freeze policy 全冻结策略
            if freeze_vit:
                for p in self.vit.parameters():
                    p.requires_grad = False

            # Optional bias-only tuning
            if bias_tune:
                for n, p in self.vit.named_parameters():
                    if 'bias' in n:
                        p.requires_grad = True

            # Optional partial layer unfreeze: unfreeze last N encoder layers
            if partial_layers and partial_layers > 0:
                try:
                    total = len(self.vit.transformer.encoder.layer)
                    for i in range(total - partial_layers, total):
                        for p in self.vit.transformer.encoder.layer[i].parameters():
                            p.requires_grad = True
                    # keep layernorm trainable for stability
                    for p in self.vit.transformer.encoder.encoder_norm.parameters():
                        p.requires_grad = True
                except Exception:
                    # if structure differs (e.g., Swin), skip
                    pass

    def forward(self, cxr_imgs: torch.Tensor, ehr_prompts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: cxr_feat (B, D), encoded sequence (B, 1+P+N, D)
        ehr_prompts: (B, P, Dv)
        """
        if self.use_hf:
            # HF 包装器路径
            cls_feat, encoded = self.vit(cxr_imgs, ehr_prompts)
        else:
            # vpt-main 路径（现有逻辑）
            # embeddings: (B, 1 + n_patches, D)
            x = self.vit.transformer.embeddings(cxr_imgs)
            B = x.shape[0]
            # Insert prompts after CLS
            # CLS Token (B, 1, D) + EHR Prompts (B, P, Dv) + Image Patches (B, N, D)
            x = torch.cat((x[:, :1, :], ehr_prompts, x[:, 1:, :]), dim=1)
            encoded, _ = self.vit.transformer.encoder(x)
            cls_feat = encoded[:, 0]
        
        return cls_feat, encoded




class CrossVPTModule(BaseFuseTrainer):
    """Lightning module: EHR -> prompts -> inject into ViT(CXR) -> heads & fusion.

    hparams requirements (from YAML):
    - hidden_size, ehr_n_head, ehr_n_layers_distinct, ehr_dropout
    - num_prompt_tokens, prompt_project_dim, prompt_dropout
    - vpt_feature, vpt_model_root, vpt_crop_size, freeze_vit, bias_tune, partial_layers
    - fusion_method (unused here; using simple concat head), loss_multi, loss_ehr, loss_cxr
    - task, class_names
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Task & dims
        self.class_names = self.hparams['class_names']
        self.num_classes = 1 if self.hparams.task == 'mortality' else len(self.class_names)

        # Build ViT for CXR
        self.cxr_vit = CXRDynamicPromptViT(
            vit_feature=self.hparams.get('vpt_feature', 'sup_vitb16_imagenet21k'),
            vit_model_root=self.hparams.get('vpt_model_root', ''),
            crop_size=self.hparams.get('vpt_crop_size', 224),
            load_pretrain=True,
            freeze_vit=self.hparams.get('freeze_vit', True),
            bias_tune=self.hparams.get('bias_tune', False),
            partial_layers=self.hparams.get('partial_layers', 0),
            hf_model_id=self.hparams.get('hf_model_id', None),  # 新增参数
        )
        vit_hidden_dim = self.cxr_vit.vit_hidden_dim
        
        # get loss weights
        self.w_multi = self.hparams.get('loss_multi', 1.0)
        self.w_ehr = self.hparams.get('aux_ehr_weight', 0.0)
        print(f"w_multi: {self.w_multi}, w_ehr: {self.w_ehr}")
        # EHR -> prompts
        ehr_input_size = 24  # consistent with existing usage
        # 构建 hparams 字典传递给 EHRPromptGenerator
        ehr_prompt_hparams = {
            'prompt_noise_std': self.hparams.get('prompt_noise_std', 0.0),
            'prompt_token_dropout': self.hparams.get('prompt_token_dropout', 0.0),
        }
        self.ehr_prompt_gen = EHRPromptGenerator(
            ehr_input_size=ehr_input_size,
            ehr_hidden_size=self.hparams.hidden_size,
            num_classes=self.num_classes,
            num_prompt_tokens=self.hparams.get('num_prompt_tokens', 5),
            vit_hidden_dim=vit_hidden_dim,
            ehr_n_head=self.hparams.get('ehr_n_head', 4),
            ehr_n_layers_distinct=self.hparams.get('ehr_n_layers_distinct', 1),
            ehr_dropout=self.hparams.get('ehr_dropout', 0.2),
            prompt_project_dim=self.hparams.get('prompt_project_dim', -1),
            prompt_dropout=self.hparams.get('prompt_dropout', 0.0),
            simple=True,
            w_ehr=self.w_ehr,  # 不传则默认 0.0 → ehr_head=None → pred_ehr 恒为 None
            hparams=ehr_prompt_hparams,  # ✨ 新增：传递正则化参数
        )

        # Heads（默认：CLS+EHR concat）
        self.cxr_head = nn.Linear(vit_hidden_dim, self.num_classes)
        self.fusion_head = nn.Linear(vit_hidden_dim + self.hparams.hidden_size, self.num_classes)

        # Variants configuration
        self.fusion_variant = self.hparams.get('fusion_variant', None)  # 'concat_avgpool' | 'cls_residual' | None
        # pool_include: 指定要包含的 tokens 类型，可选 'cls', 'prompt', 'image'/'patches'
        # 如果未设置，则使用 pool_exclude 进行兼容（向后兼容）
        if 'pool_include' in self.hparams:
            pool_include_list = self.hparams.get('pool_include', ['cls', 'prompt', 'image'])
            self.pool_include = set(pool_include_list)
        else:
            # 向后兼容：从 pool_exclude 转换为 pool_include
            pool_exclude = set(self.hparams.get('pool_exclude', ['cls', 'prompt']))
            all_options = {'cls', 'prompt', 'image'}
            # 将 'patches' 也视为 'image'
            if 'patches' in pool_exclude:
                pool_exclude.add('image')
                pool_exclude.discard('patches')
            self.pool_include = all_options - pool_exclude
        self.num_prompt_tokens = self.hparams.get('num_prompt_tokens', 5)

        # concat_avgpool extra heads (same structure as Late fusion: direct concat + classification)
        if self.fusion_variant == 'concat_avgpool':
            in_dim = vit_hidden_dim + self.hparams.hidden_size  # 两个模态特征拼接的维度
            self.fusion_head_avg = nn.Linear(in_dim, self.num_classes)

        # cls_residual components
        if self.fusion_variant == 'cls_residual':
            self.ehr2vit = nn.Linear(self.hparams.hidden_size, vit_hidden_dim)
            alpha_init = float(self.hparams.get('alpha_init', 0.1))
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # Loss
        self.pred_criterion = nn.BCELoss()

    def forward(self, batch):
        # Inputs
        ehr_ts = batch['ehr_ts']  # (B, T, E)
        seq_len = batch['seq_len']
        cxr_imgs = batch['cxr_imgs']  # (B, 3, H, W)
        labels = batch['labels']

        # EHR enc -> prompts
        ehr_feat, ehr_prompts, pred_ehr = self.ehr_prompt_gen(ehr_ts, seq_len)

        # Inject into ViT（返回 CLS 和完整序列）
        cxr_feat, encoded = self.cxr_vit(cxr_imgs, ehr_prompts)
        pred_cxr = torch.sigmoid(self.cxr_head(cxr_feat))

        # 优先使用 fusion_variant 分支；否则回退到 final_pred_mode
        variant = self.fusion_variant
        if variant == 'concat_avgpool':
            # 取出指定的 tokens 做全局平均池化
            # 序列结构：[CLS (0)] + [Prompts (1 to P)] + [Image Patches (P+1 to P+N)]
            # 其中 P = num_prompt_tokens
            tokens_list = []
            
            # 根据 pool_include 拼接对应的 tokens
            if 'cls' in self.pool_include:
                tokens_list.append(encoded[:, 0:1, :])  # CLS token: (B, 1, D)
            
            if 'prompt' in self.pool_include:
                prompt_start = 1
                prompt_end = 1 + self.num_prompt_tokens
                tokens_list.append(encoded[:, prompt_start:prompt_end, :])  # Prompt tokens: (B, P, D)
            
            if 'image' in self.pool_include or 'patches' in self.pool_include:
                image_start = 1 + self.num_prompt_tokens
                tokens_list.append(encoded[:, image_start:, :])  # Image patches: (B, N, D)
            
            if len(tokens_list) == 0:
                raise ValueError("pool_include 不能为空，至少需要包含 'cls', 'prompt', 或 'image'/'patches' 之一")
            
            # 拼接所有包含的 tokens
            if len(tokens_list) == 1:
                img_tokens = tokens_list[0]
            else:
                img_tokens = torch.cat(tokens_list, dim=1)  # (B, sum_of_lengths, D)
            
            included_parts = sorted(list(self.pool_include))
            print(f"pool_include: {', '.join(included_parts)}")
            
            img_avg = img_tokens.mean(dim=1)
            concat = torch.cat([img_avg, ehr_feat], dim=1)
            pred_multi = torch.sigmoid(self.fusion_head_avg(concat))

            w_multi = self.hparams.get('loss_multi', 1.0)
            w_ehr = self.hparams.get('loss_ehr', 1.0)
            w_cxr = self.hparams.get('loss_cxr', 1.0)

            loss_multi = self.pred_criterion(pred_multi, labels)
            loss_ehr = self.pred_criterion(pred_ehr, labels)
            loss_cxr = self.pred_criterion(pred_cxr, labels)
            loss = w_multi * loss_multi + w_ehr * loss_ehr + w_cxr * loss_cxr

        elif variant == 'cls_residual':
            # 在 CLS 上做残差融合
            ehr_proj = self.ehr2vit(ehr_feat)
            fusion_cls = cxr_feat + self.alpha * ehr_proj
            pred_multi = torch.sigmoid(self.cxr_head(fusion_cls))

            w_multi = self.hparams.get('loss_multi', 1.0)
            w_ehr = self.hparams.get('aux_ehr_weight', 0.0)
            w_cxr = self.hparams.get('loss_cxr', 0.0)  # 可选：对单 CXR 分支加监督，避免单模态指标退化
            loss_multi = self.pred_criterion(pred_multi, labels)
            loss_ehr = self.pred_criterion(pred_ehr, labels)
            loss_cxr = self.pred_criterion(pred_cxr, labels)
            loss = w_multi * loss_multi + w_ehr * loss_ehr + w_cxr * loss_cxr

        else:
            final_mode = self.hparams.get('final_pred_mode', 'concat')
            if final_mode == 'cxr_only':
                pred_multi = pred_cxr
                w_multi = self.hparams.get('loss_multi', 1.0)
                w_ehr = self.hparams.get('aux_ehr_weight', 0.0)
        
                loss_multi = self.pred_criterion(pred_multi, labels)
                loss_ehr = self.pred_criterion(pred_ehr, labels)
                loss = w_multi * loss_multi + w_ehr * loss_ehr
            else:
                fused = torch.cat([ehr_feat, cxr_feat], dim=1)
                pred_multi = torch.sigmoid(self.fusion_head(fused))
                w_multi = self.hparams.get('loss_multi', 1.0)
                w_ehr = self.hparams.get('loss_ehr', 1.0)
                w_cxr = self.hparams.get('loss_cxr', 1.0)
                loss_multi = self.pred_criterion(pred_multi, labels)
                loss_ehr = self.pred_criterion(pred_ehr, labels)
                loss_cxr = self.pred_criterion(pred_cxr, labels)
                loss = w_multi * loss_multi + w_ehr * loss_ehr + w_cxr * loss_cxr

        out = {
            'predictions': pred_multi,
            'pred_ehr': pred_ehr,
            'pred_cxr': pred_cxr,
            'feat_ehr_distinct': ehr_feat,
            'feat_cxr_distinct': cxr_feat,
            'loss': loss,
        }
        return out
