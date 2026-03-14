# Model Hyperparameters and Configurations

This document provides a comprehensive overview of hyperparameter search spaces and best configurations for all models in CareBench. The hyperparameters were optimized using Bayesian optimization across different tasks (mortality, phenotype, length of stay) and cohorts (base cohort and matched subset).

## Hyperparameter Search Spaces Overview

| Model | Hyperparameter Search Space |
|-------|----------------------------|
| **DrFuse** | $\lambda_{\text{disentangle\_shared}}, \lambda_{\text{disentangle\_ehr}}, \lambda_{\text{disentangle\_cxr}}, \lambda_{\text{pred\_ehr}}, \lambda_{\text{pred\_cxr}}, \lambda_{\text{pred\_shared}}, \lambda_{\text{attn\_aux}} \in [0.01, 2.0]$ |
| **FlexMoE** | $num_{\text{experts}} \in \{4,8,16\}, num_{\text{routers}} \in \{1,2\}, k \in \{2,4,8\}, \lambda_{\text{gate}} \in [0.001, 0.1]$ |
| **HEALNet** | $f_{\text{depth}} \in \{1,2,3\}, N_{\text{freq}} \in \{1,2,4\}, f_{\text{max}} \in \{5.0, 10.0\}$ |
| **M3Care** | $\lambda_{\text{stab\_reg}} \in [0.001, 2.0]$ |
| **ShaSpec** | $\alpha \in [0.01,0.1], \beta \in [0.005,0.2]$ |
| **SMIL** | $T_{\text{inner}} \in \{1,2,3\}, M_{\text{MC}} \in \{10,20,30\}, \eta_{\text{inner}} \in [10^{-4},10^{-3}], \alpha_{\text{feat}} \in [0.05,0.2], \beta_{\text{ehr}} \in [0.05,0.2], \tau \in [1.0,3.0]$ |
| **AUG** | $\alpha_{\text{merge}} \in [0.2, 0.8], \lambda \in [0.8, 1.2], \delta \text{ (margin)} \in [0.005, 0.05], T_{\text{check interval}} \in \{5,10,15\}$ |
| **InfoReg** | $k_{\text{threshold}} \in [0.01, 0.10], \gamma_{\text{ehr}} \in [0.5, 1.5], \gamma_{\text{cxr}} \in [0.05, 0.50], T_{\text{history}} \in \{5,10,15\}$ |

**Note**: General training parameters (learning rate, batch size, epochs, patience, seeds) are fixed across all models.

---

## DrFuse Configuration

Full configuration of DrFuse with best hyperparameters from Bayesian optimization across tasks and cohorts.

| Category | Parameter | Value / Best value |
|----------|-----------|---------------------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | EHR heads | 4 |
| | EHR layers (distinct/feat/shared) | 1 / 1 / 1 |
| | EHR hidden size | 256 |
| | CXR encoder | ResNet-50 |
| **Fusion (fixed)** | Fusion method | concatenate |
| | Logit average | true |
| | Attention fusion | mid |
| | Disentangle loss | jsd |
| **Phenotype (Base cohort)** | $\lambda_{\text{disentangle\_shared}}$ | 0.01 |
| | $\lambda_{\text{disentangle\_ehr}}$ | 0.7626 |
| | $\lambda_{\text{disentangle\_cxr}}$ | 2.0 |
| | $\lambda_{\text{pred\_ehr}}$ | 2.0 |
| | $\lambda_{\text{pred\_cxr}}$ | 2.0 |
| | $\lambda_{\text{pred\_shared}}$ | 2.0 |
| | $\lambda_{\text{attn\_aux}}$ | 1.8578 |
| **Phenotype (Matched subset)** | $\lambda_{\text{disentangle\_shared}}$ | 0.4796 |
| | $\lambda_{\text{disentangle\_ehr}}$ | 0.5195 |
| | $\lambda_{\text{disentangle\_cxr}}$ | 0.0904 |
| | $\lambda_{\text{pred\_ehr}}$ | 1.4242 |
| | $\lambda_{\text{pred\_cxr}}$ | 0.2306 |
| | $\lambda_{\text{pred\_shared}}$ | 0.8842 |
| | $\lambda_{\text{attn\_aux}}$ | 0.4114 |
| **Mortality (Base cohort)** | $\lambda_{\text{disentangle\_shared}}$ | 0.01 |
| | $\lambda_{\text{disentangle\_ehr}}$ | 0.8112 |
| | $\lambda_{\text{disentangle\_cxr}}$ | 0.8074 |
| | $\lambda_{\text{pred\_ehr}}$ | 2.0 |
| | $\lambda_{\text{pred\_cxr}}$ | 1.5418 |
| | $\lambda_{\text{pred\_shared}}$ | 1.0930 |
| | $\lambda_{\text{attn\_aux}}$ | 0.0164 |
| **Mortality (Matched subset)** | $\lambda_{\text{disentangle\_shared}}$ | 1.8540 |
| | $\lambda_{\text{disentangle\_ehr}}$ | 1.4572 |
| | $\lambda_{\text{disentangle\_cxr}}$ | 0.6598 |
| | $\lambda_{\text{pred\_ehr}}$ | 1.1451 |
| | $\lambda_{\text{pred\_cxr}}$ | 1.046 |
| | $\lambda_{\text{pred\_shared}}$ | 1.9227 |
| | $\lambda_{\text{attn\_aux}}$ | 1.6906 |
| **LoS (Base cohort)** | $\lambda_{\text{disentangle\_shared}}$ | 0.0115 |
| | $\lambda_{\text{disentangle\_ehr}}$ | 1.9845 |
| | $\lambda_{\text{disentangle\_cxr}}$ | 1.2387 |
| | $\lambda_{\text{pred\_ehr}}$ | 1.2271 |
| | $\lambda_{\text{pred\_cxr}}$ | 0.0240 |
| | $\lambda_{\text{pred\_shared}}$ | 0.0558 |
| | $\lambda_{\text{attn\_aux}}$ | 1.0543 |
| **LoS (Matched subset)** | $\lambda_{\text{disentangle\_shared}}$ | 1.8540 |
| | $\lambda_{\text{disentangle\_ehr}}$ | 1.4572 |
| | $\lambda_{\text{disentangle\_cxr}}$ | 0.6598 |
| | $\lambda_{\text{pred\_ehr}}$ | 1.1451 |
| | $\lambda_{\text{pred\_cxr}}$ | 1.0464 |
| | $\lambda_{\text{pred\_shared}}$ | 1.9227 |
| | $\lambda_{\text{attn\_aux}}$ | 1.6906 |

---

## HEALNet Configuration

Full configuration of HEALNet with best hyperparameters from Bayesian optimization across tasks and cohorts.

| Category | Parameter | Value / Best value |
|----------|-----------|---------------------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | N_modalities | 2 (EHR + CXR) |
| | Latent channels | 256 |
| | Latent dimension | 256 |
| | Cross-attention heads | 4 |
| | Latent attention heads | 4 |
| | Cross head dimension | 64 |
| | Latent head dimension | 64 |
| | Self per cross attention | 1 |
| | Weight tie layers | true |
| | Self-normalizing nets | true |
| | Fourier encoding | true |
| | Final classifier head | true |
| | Attention dropout | 0.2 |
| | Feed-forward dropout | 0.2 |
| **Phenotype (Base cohort)** | Fusion depth | 1 |
| | Num frequency bands | 4 |
| | Max frequency | 10 |
| **Phenotype (Matched subset)** | Fusion depth | 1 |
| | Num frequency bands | 4 |
| | Max frequency | 5 |
| **Mortality (Base cohort)** | Fusion depth | 1 |
| | Num frequency bands | 4 |
| | Max frequency | 5 |
| **Mortality (Matched subset)** | Fusion depth | 3 |
| | Num frequency bands | 2 |
| | Max frequency | 5 |
| **LoS (Base cohort)** | Fusion depth | 3 |
| | Num frequency bands | 1 |
| | Max frequency | 10 |
| **LoS (Matched subset)** | Fusion depth | 2 |
| | Num frequency bands | 2 |
| | Max frequency | 5 |

---

## M3Care Configuration

Full configuration of M3Care with best hyperparameters from Bayesian optimization across tasks and cohorts.

| Category | Parameter | Value / Best value |
|----------|-----------|---------------------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 |
| **Architecture (fixed)** | Hidden dimension | 256 |
| | EHR attention heads | 4 |
| | EHR layers | 1 |
| | Max sequence length | 500 |
| | LSTM bidirectional | true |
| | LSTM layers | 1 |
| **Search (M3Care-specific)** | $\lambda_{\text{stab\_reg}}$ | [0.001, 2.0] |
| **Phenotype (Base cohort)** | $\lambda_{\text{stab\_reg}}$ | 0.001 |
| **Phenotype (Matched subset)** | $\lambda_{\text{stab\_reg}}$ | 0.001 |
| **Mortality (Base cohort)** | $\lambda_{\text{stab\_reg}}$ | 0.001 |
| **Mortality (Matched subset)** | $\lambda_{\text{stab\_reg}}$ | 1.5932 |
| **LoS (Base cohort)** | $\lambda_{\text{stab\_reg}}$ | 0.1865 |
| **LoS (Matched subset)** | $\lambda_{\text{stab\_reg}}$ | 0.7189 |

---

## AUG Configuration

Full configuration of AUG with best hyperparameters across tasks and cohorts.

| Category | Parameter | Value / Best value |
|----------|-----------|---------------------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 (pretrained) |
| **Architecture (fixed)** | Hidden dimension | 256 |
| | EHR attention heads | 4 |
| | EHR layers | 1 |
| | EHR dropout | 0.2 |
| | Max sequence length | 500 |
| **Phenotype (Base cohort)** | Fusion weight $\alpha_{\text{merge}}$ | 0.7225 |
| | Target ratio $\lambda$ | 1.2000 |
| | Margin $\delta$ | 0.0050 |
| | Layer check interval $T_{\text{check}}$ | 15 |
| **Phenotype (Matched subset)** | Fusion weight $\alpha_{\text{merge}}$ | 0.6484 |
| | Target ratio $\lambda$ | 1.0159 |
| | Margin $\delta$ | 0.0314 |
| | Layer check interval $T_{\text{check}}$ | 15 |
| **Mortality (Base cohort)** | Fusion weight $\alpha_{\text{merge}}$ | 0.6050 |
| | Target ratio $\lambda$ | 1.1852 |
| | Margin $\delta$ | 0.0422 |
| | Layer check interval $T_{\text{check}}$ | 10 |
| **Mortality (Matched subset)** | Fusion weight $\alpha_{\text{merge}}$ | 0.7360 |
| | Target ratio $\lambda$ | 1.2000 |
| | Margin $\delta$ | 0.0400 |
| | Layer check interval $T_{\text{check}}$ | 15 |
| **LoS (Base cohort)** | Fusion weight $\alpha_{\text{merge}}$ | 0.6484 |
| | Target ratio $\lambda$ | 1.0159 |
| | Margin $\delta$ | 0.0314 |
| | Layer check interval $T_{\text{check}}$ | 15 |
| **LoS (Matched subset)** | Fusion weight $\alpha_{\text{merge}}$ | 0.7989 |
| | Target ratio $\lambda$ | 0.9836 |
| | Margin $\delta$ | 0.0297 |
| | Layer check interval $T_{\text{check}}$ | 10 |

---

## InfoReg Configuration

Full configuration of InfoReg with best hyperparameters across tasks and cohorts.

| Category | Parameter | Value / Best value |
|----------|-----------|---------------------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 (pretrained) |
| **Architecture (fixed)** | Hidden dimension | 256 |
| | EHR attention heads | 4 |
| | EHR layers | 1 |
| | EHR dropout | 0.2 |
| | Max sequence length | 500 |
| **Phenotype (Base cohort)** | Threshold $k_{\text{thr}}$ | 0.0687 |
| | EHR scale $\gamma_{\text{ehr}}$ | 1.2161 |
| | CXR scale $\gamma_{\text{cxr}}$ | 0.2750 |
| | History length $T_{\text{hist}}$ | 10 |
| **Phenotype (Matched subset)** | Threshold $k_{\text{thr}}$ | 0.0723 |
| | EHR scale $\gamma_{\text{ehr}}$ | 0.6116 |
| | CXR scale $\gamma_{\text{cxr}}$ | 0.0527 |
| | History length $T_{\text{hist}}$ | 10 |
| **Mortality (Base cohort)** | Threshold $k_{\text{thr}}$ | 0.0102 |
| | EHR scale $\gamma_{\text{ehr}}$ | 0.5117 |
| | CXR scale $\gamma_{\text{cxr}}$ | 0.4373 |
| | History length $T_{\text{hist}}$ | 15 |
| **Mortality (Matched subset)** | Threshold $k_{\text{thr}}$ | 0.0449 |
| | EHR scale $\gamma_{\text{ehr}}$ | 0.5399 |
| | CXR scale $\gamma_{\text{cxr}}$ | 0.0972 |
| | History length $T_{\text{hist}}$ | 5 |
| **LoS (Base cohort)** | Threshold $k_{\text{thr}}$ | 0.0583 |
| | EHR scale $\gamma_{\text{ehr}}$ | 1.1327 |
| | CXR scale $\gamma_{\text{cxr}}$ | 0.3866 |
| | History length $T_{\text{hist}}$ | 15 |
| **LoS (Matched subset)** | Threshold $k_{\text{thr}}$ | 0.0745 |
| | EHR scale $\gamma_{\text{ehr}}$ | 0.5021 |
| | CXR scale $\gamma_{\text{cxr}}$ | 0.0674 |
| | History length $T_{\text{hist}}$ | 15 |

---

## SMIL Configuration

Full configuration of SMIL with best hyperparameters from Bayesian optimization across tasks and cohorts.

| Category | Parameter | Value / Best value |
|----------|-----------|---------------------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 |
| **Architecture (fixed)** | Hidden dimension | 256 |
| | EHR attention heads | 4 |
| | EHR layers | 1 |
| | Max sequence length | 500 |
| | Number of clusters | 10 |
| **Phenotype (Base cohort)** | Inner loop iterations | 2 |
| | Monte Carlo size | 20 |
| | Inner learning rate | 0.0007 |
| | $\alpha$ | 0.05 |
| | $\beta$ | 0.0882 |
| | Temperature | 3.0 |
| **Phenotype (Matched subset)** | Inner loop iterations | 1 |
| | Monte Carlo size | 20 |
| | Inner learning rate | 0.0008 |
| | $\alpha$ | 0.052 |
| | $\beta$ | 0.091 |
| | Temperature | 2.6473 |
| **Mortality (Base cohort)** | Inner loop iterations | 2 |
| | Monte Carlo size | 10 |
| | Inner learning rate | 0.0005 |
| | $\alpha$ | 0.1460 |
| | $\beta$ | 0.1829 |
| | Temperature | 2.1659 |
| **Mortality (Matched subset)** | Inner loop iterations | 2 |
| | Monte Carlo size | 20 |
| | Inner learning rate | 0.0004 |
| | $\alpha$ | 0.141 |
| | $\beta$ | 0.176 |
| | Temperature | 2.2237 |
| **LoS (Base cohort)** | Inner loop iterations | 3 |
| | Monte Carlo size | 20 |
| | Inner learning rate | 0.0002 |
| | $\alpha$ | 0.0569 |
| | $\beta$ | 0.1960 |
| | Temperature | 1.4655 |
| **LoS (Matched subset)** | Inner loop iterations | 3 |
| | Monte Carlo size | 10 |
| | Inner learning rate | 0.0004 |
| | $\alpha$ | 0.1099 |
| | $\beta$ | 0.1755 |
| | Temperature | 1.0842 |

---

## DAFT Configuration

Full configuration of DAFT. No hyperparameter search was performed as all parameters are fixed.

| Category | Parameter | Value |
|----------|-----------|-------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Seeds | {42, 123, 1234} |
| | Dropout | 0.2 |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 |
| | EHR attention heads | 4 |
| | EHR layers | 1 |
| **DAFT fusion (fixed)** | Layer after | -1 (all layers) |
| | Activation | linear |
| **Architecture (fixed)** | Hidden dimension | 256 |

---

## FlexMoE Configuration

Full configuration of FlexMoE with best hyperparameters from Bayesian optimization across tasks and cohorts.

| Category | Parameter | Value / Best value |
|----------|-----------|---------------------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 |
| **Architecture (fixed)** | Hidden dimension | 256 |
| | Num patches | 16 |
| | Num layers | 1 |
| | Num prediction layers | 1 |
| | Num heads | 4 |
| **EHR Transformer (fixed)** | Attention heads | 4 |
| | Layers | 1 |
| **Phenotype (Base cohort)** | Num experts | 8 |
| | Num routers | 2 |
| | Top-$k$ | 4 |
| | Gate loss weight | 0.1 |
| **Phenotype (Matched subset)** | Num experts | 8 |
| | Num routers | 2 |
| | Top-$k$ | 4 |
| | Gate loss weight | 0.01 |
| **Mortality (Base cohort)** | Num experts | 16 |
| | Num routers | 1 |
| | Top-$k$ | 2 |
| | Gate loss weight | 0.001 |
| **Mortality (Matched subset)** | Num experts | 16 |
| | Num routers | 1 |
| | Top-$k$ | 8 |
| | Gate loss weight | 0.0591 |
| **LoS (Base cohort)** | Num experts | 8 |
| | Num routers | 2 |
| | Top-$k$ | 2 |
| | Gate loss weight | 0.0994 |
| **LoS (Matched subset)** | Num experts | 4 |
| | Num routers | 2 |
| | Top-$k$ | 2 |
| | Gate loss weight | 0.001 |

---

## LateFusion Configuration

Full configuration of LateFusion. No hyperparameter search was performed as all parameters are fixed.

| Category | Parameter | Value |
|----------|-----------|-------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 |
| **Architecture (fixed)** | Hidden size | 256 |
| | EHR layers | 1 |
| | EHR attention heads | 4 |
| | EHR dropout | 0.2 |

---

## ShaSpec Configuration

Full configuration of ShaSpec with best hyperparameters from Bayesian optimization across tasks and cohorts.

| Category | Parameter | Value / Best value |
|----------|-----------|---------------------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 |
| **Architecture (fixed)** | Hidden dimension | 256 |
| | Weight standardization | true |
| | EHR attention heads | 4 |
| | EHR layers | 1 |
| | Shared transformer heads | 4 |
| | Shared transformer layers | 1 |
| | Max sequence length | 500 |
| **Phenotype (Base cohort)** | $\alpha$ | 0.01 |
| | $\beta$ | 0.1608 |
| **Phenotype (Matched subset)** | $\alpha$ | 0.01 |
| | $\beta$ | 0.0217 |
| **Mortality (Base cohort)** | $\alpha$ | 0.0261 |
| | $\beta$ | 0.0283 |
| **Mortality (Matched subset)** | $\alpha$ | 0.0530 |
| | $\beta$ | 0.05759 |
| **LoS (Base cohort)** | $\alpha$ | 0.0539 |
| | $\beta$ | 0.0240 |
| **LoS (Matched subset)** | $\alpha$ | 0.0816 |
| | $\beta$ | 0.0407 |

---

## LSTM Configuration

Full configuration of the LSTM. No hyperparameter search was performed as all parameters are fixed.

| Category | Parameter | Value |
|----------|-----------|-------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Architecture (fixed)** | Hidden size | 256 |
| | Num layers | 1 |
| | Bidirectional | true |
| | Dropout | 0.2 |

---

## MedFuse Configuration

Full configuration of MedFuse. No hyperparameter search was performed as all parameters are fixed.

| Category | Parameter | Value |
|----------|-----------|-------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | LSTM |
| | CXR encoder | ResNet-50 |
| | EHR LSTM bidirectional | true |
| **Architecture (fixed)** | Hidden dimension | 256 |
| | LSTM layers | 1 |
| | Fusion type | LSTM |

---

## MMTM Configuration

Full configuration of MMTM. No hyperparameter search was performed as all parameters are fixed.

| Category | Parameter | Value |
|----------|-----------|-------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 |
| **Architecture (fixed)** | Hidden dimension | 256 |
| | EHR attention heads | 4 |
| | EHR layers | 1 |
| **MMTM Fusion (fixed)** | Compression ratio | 4 |
| | Layer after | -1 (all layers) |

---

## ResNet Configuration

Full configuration of the ResNet. No hyperparameter search was performed as all parameters are fixed.

| Category | Parameter | Value |
|----------|-----------|-------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Architecture (fixed)** | Hidden size | 256 |

---

## UMSE Configuration

Full configuration of UMSE. No hyperparameter search was performed as all parameters are fixed.

| Category | Parameter | Value |
|----------|-----------|-------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Architecture (fixed)** | Model dimension | 256 |
| | Transformer layers | 1 |
| | Attention heads | 4 |
| **Fusion (fixed)** | Bottlenecks (MBT) | 1 |

---

## UTDE Configuration

Full configuration of UTDE. No hyperparameter search was performed as all parameters are fixed.

| Category | Parameter | Value |
|----------|-----------|-------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Encoder (fixed)** | EHR encoder | Transformer |
| | CXR encoder | ResNet-50 |
| **Architecture (fixed)** | Embedding dimension | 256 |
| | EHR num layers | 1 |
| | EHR attention heads | 4 |
| | Time embedding dimension | 64 |
| | Transformer attention heads | 4 |
| | Cross-modal layers | 1 |
| | Max EHR sequence length | 500 |

---

## Transformer Configuration

Full configuration of the Transformer baseline. No hyperparameter search was performed as all parameters are fixed.

| Category | Parameter | Value |
|----------|-----------|-------|
| **General (fixed)** | Learning rate | 0.0001 |
| | Batch size | 16 |
| | Epochs | 50 |
| | Patience | 10 |
| | Dropout | 0.2 |
| | Seeds | {42, 123, 1234} |
| **Architecture (fixed)** | Model dimension | 256 |
| | Transformer layers | 1 |
| | Attention heads | 4 |
