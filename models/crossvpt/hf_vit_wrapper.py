"""
HuggingFace ViT Wrapper for CheXpert Pre-trained Models

This module provides wrapper classes for HuggingFace Vision Transformer models,
enabling integration with the CrossVPT and LateFusion frameworks.

Supports:
- Dynamic prompt injection (for CrossVPT)
- Static feature extraction (for LateFusion)
- Flexible freezing strategies (full freeze, bias-only, partial layers)
"""

import torch
from torch import nn
from typing import Tuple, Optional
from transformers import AutoModelForImageClassification


class CXRDynamicPromptViT_HF(nn.Module):
    """
    HuggingFace ViT 包装器，支持 EHR prompt 注入（用于 CrossVPT）
    
    架构流程：
    1. CXR 图像 → vit.embeddings() → [CLS + patches] + position + dropout
    2. 注入 prompt: [CLS, prompts, patches]
    3. vit.encoder() → Transformer encoding
    4. vit.layernorm() → Layer normalization（关键！）
    5. 提取 CLS: (B, 768)
    
    Args:
        hf_model_id: HuggingFace 模型 ID，默认 "codewithdark/vit-chest-xray"
        freeze_vit: 是否冻结 ViT 参数
        bias_tune: 是否仅微调 bias 参数
        partial_layers: 解冻最后 N 层 encoder
    """
    
    def __init__(
        self,
        hf_model_id: str = "codewithdark/vit-chest-xray",
        freeze_vit: bool = True,
        bias_tune: bool = False,
        partial_layers: int = 0,
    ):
        super().__init__()
        
        # 加载 HF 模型（带验证信息）
        print(f"\n{'='*60}")
        print(f"[HF ViT] 正在加载 HuggingFace 预训练模型...")
        print(f"  模型 ID: {hf_model_id}")
        print(f"{'='*60}\n")
        
        self.model = AutoModelForImageClassification.from_pretrained(
            hf_model_id,
            ignore_mismatched_sizes=True  # 允许分类头维度不匹配
        )
        
        # 打印模型信息以验证预训练权重
        print(f"\n{'='*60}")
        print(f"[HF ViT] ✅ 模型加载完成！验证信息：")
        print(f"{'='*60}")
        print(f"  模型架构：{self.model.__class__.__name__}")
        print(f"  模型类型：{self.model.config.model_type}")
        print(f"  隐藏层维度：{self.model.config.hidden_size}")
        print(f"  Transformer 层数：{self.model.config.num_hidden_layers}")
        print(f"  注意力头数：{self.model.config.num_attention_heads}")
        print(f"  图像尺寸：{self.model.config.image_size}")
        print(f"  Patch 大小：{self.model.config.patch_size}")
        print(f"  词汇表大小：{self.model.config.num_labels}")
        
        # 检查是否成功加载预训练权重
        if hasattr(self.model, 'name_or_path'):
            print(f"  模型来源：{self.model.name_or_path}")
        
        # 检查预训练任务
        if hasattr(self.model.config, 'id2label'):
            print(f"  预训练任务标签数：{len(self.model.config.id2label)}")
        
        print(f"{'='*60}\n")
        
        self.vit_hidden_dim = self.model.config.hidden_size  # 768 for ViT-Base
        
        # 应用冻结/微调策略
        self._apply_freeze_strategy(freeze_vit, bias_tune, partial_layers)
    
    def _apply_freeze_strategy(self, freeze_vit: bool, bias_tune: bool, partial_layers: int):
        """应用冻结/微调策略"""
        
        # 处理字符串到布尔值的转换（命令行参数可能是字符串）
        if isinstance(freeze_vit, str):
            freeze_vit = freeze_vit.lower() == 'true'
        if isinstance(bias_tune, str):
            bias_tune = bias_tune.lower() == 'true'
        if isinstance(partial_layers, str):
            partial_layers = int(partial_layers)
        # 处理 None 值（默认值）
        if partial_layers is None:
            partial_layers = 0
        
        # 验证预训练权重（检查第一层权重是否接近非随机分布）
        if freeze_vit or (not bias_tune and partial_layers == 0):
            # 只在完全冻结或完全微调模式下验证
            first_layer_weight = self.model.vit.encoder.layer[0].attention.attention.query.weight
            weight_std = first_layer_weight.std().item()
            weight_mean = first_layer_weight.mean().item()
            
            # 预训练模型的权重通常 std 在 0.02-0.5 之间，而不是随机初始化的 1.0 左右
            is_pretrained = 0.01 < weight_std < 0.8
            print(f"\n{'='*60}")
            print(f"[HF ViT] 预训练权重验证：")
            print(f"  第一层权重统计 - Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")
            if is_pretrained:
                print(f"  ✅ 确认：已加载预训练权重（非随机初始化）")
            else:
                print(f"  ⚠️ 警告：权重统计异常，可能未正确加载预训练权重")
            print(f"{'='*60}\n")
        
        # 统计参数
        total_params = 0
        trainable_params = 0
        
        if freeze_vit:
            # 完全冻结：关闭 dropout，确保特征稳定
            self.model.vit.eval()
            for p in self.model.vit.parameters():
                p.requires_grad = False
        
        if bias_tune:
            # Bias-only 微调：仅解冻 bias 参数
            for n, p in self.model.vit.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        
        if partial_layers > 0:
            # 解冻最后 N 层 encoder + layernorm
            total_layers = len(self.model.vit.encoder.layer)
            print(f"[HF ViT] Unfreezing last {partial_layers}/{total_layers} encoder layers")
            
            for i in range(total_layers - partial_layers, total_layers):
                layer = self.model.vit.encoder.layer[i]
                layer.train()  # 开启 dropout
                for p in layer.parameters():
                    p.requires_grad = True
            
            # 解冻 layernorm
            self.model.vit.layernorm.train()
            for p in self.model.vit.layernorm.parameters():
                p.requires_grad = True
        
        # 统计可训练参数
        for n, p in self.model.vit.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        
        # 打印冻结策略信息
        print(f"\n{'='*60}")
        print(f"[HF ViT Wrapper] 冻结策略统计")
        print(f"{'='*60}")
        print(f"  模型：{self.model.config.model_type}")
        print(f"  隐藏层维度：{self.model.config.hidden_size}")
        print(f"  Transformer 层数：{self.model.config.num_hidden_layers}")
        print(f"  注意力头数：{self.model.config.num_attention_heads}")
        print(f"\n  冻结配置:")
        print(f"    freeze_vit: {freeze_vit}")
        print(f"    bias_tune: {bias_tune}")
        print(f"    partial_layers: {partial_layers}")
        print(f"\n  参数量统计:")
        print(f"    总参数量：{total_params:,} ({total_params/1e6:.2f}M)")
        print(f"    可训练参数：{trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"    冻结参数：{total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
        print(f"    可训练比例：{trainable_params/total_params*100:.2f}%")
        
        if freeze_vit and trainable_params == 0:
            print(f"\n  ✅ ViT 已完全冻结")
        elif freeze_vit and trainable_params > 0:
            print(f"\n  ⚠️  ViT 部分解冻 (bias_tune={bias_tune} 或 partial_layers={partial_layers})")
        else:
            print(f"\n  ✅ ViT 全参数可训练 (fine-tuning)")
        print(f"{'='*60}\n")
    
    def forward(
        self, 
        cxr_imgs: torch.Tensor, 
        ehr_prompts: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（支持 prompt 注入）
        
        Args:
            cxr_imgs: CXR 图像 (B, 3, 224, 224)
            ehr_prompts: EHR prompt tokens (B, P, 768) 或 None
        
        Returns:
            cls_feat: CLS 特征 (B, 768)
            encoded: 完整编码序列 (B, seq_len, 768)
                     seq_len = 197 (无 prompt) 或 1+P+196 (有 prompt)
        """
        # 1. 获取 embeddings（已包含位置编码 + dropout）
        embeddings = self.model.vit.embeddings(pixel_values=cxr_imgs)
        # embeddings.shape: (B, 197, 768) = [CLS + 196 patches]
        
        # 2. 可选 prompt 注入
        if ehr_prompts is not None:
            # 注入前对齐 dtype/device（混合精度训练时很重要）
            ehr_prompts = ehr_prompts.to(dtype=embeddings.dtype, device=embeddings.device)
            
            # 注入位置：CLS 之后，patches 之前
            # [CLS, prompts, patches]
            x = torch.cat((
                embeddings[:, :1, :],  # CLS: (B, 1, 768)
                ehr_prompts,            # (B, P, 768)
                embeddings[:, 1:, :]    # Patches: (B, 196, 768)
            ), dim=1)
            # x.shape: (B, 1+P+196, 768)
        else:
            x = embeddings
            # x.shape: (B, 197, 768)
        
        # 3. Transformer encoder
        # 注意：直接调用 encoder 时不使用 return_dict 参数
        # 如果输入是 tuple，则返回 tuple；如果是 Tensor，则返回 Tensor
        encoder_output = self.model.vit.encoder(x)
        
        # 处理输出：可能是 tuple 或 BaseModelOutput
        if isinstance(encoder_output, tuple):
            hidden_state = encoder_output[0]  # (B, seq_len, 768)
        elif hasattr(encoder_output, 'last_hidden_state'):
            hidden_state = encoder_output.last_hidden_state
        else:
            hidden_state = encoder_output
        
        # 4. LayerNorm（关键！HF ViT 的标准流程）
        hidden_state = self.model.vit.layernorm(hidden_state)
        
        # 5. 提取 CLS 特征
        cls_feat = hidden_state[:, 0]  # (B, 768)
        
        return cls_feat, hidden_state


class CXRStaticViT_HF(nn.Module):
    """
    HuggingFace ViT 包装器，不支持 prompt 注入（用于 LateFusion）
    
    这是 CXRDynamicPromptViT_HF 的简化版本，固定 ehr_prompts=None
    """
    
    def __init__(
        self,
        hf_model_id: str = "codewithdark/vit-chest-xray",
        freeze_vit: bool = True,
        bias_tune: bool = False,
        partial_layers: int = 0,
    ):
        super().__init__()
        
        # 内部使用 Dynamic 包装器
        self.wrapper = CXRDynamicPromptViT_HF(
            hf_model_id=hf_model_id,
            freeze_vit=freeze_vit,
            bias_tune=bias_tune,
            partial_layers=partial_layers,
        )
        self.vit_hidden_dim = self.wrapper.vit_hidden_dim
    
    def forward(
        self, 
        cxr_imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（无 prompt）
        
        Args:
            cxr_imgs: CXR 图像 (B, 3, 224, 224)
        
        Returns:
            cls_feat: CLS 特征 (B, 768)
            encoded: 完整编码序列 (B, 197, 768)
        """
        cls_feat, encoded = self.wrapper(cxr_imgs, ehr_prompts=None)
        return cls_feat, encoded
