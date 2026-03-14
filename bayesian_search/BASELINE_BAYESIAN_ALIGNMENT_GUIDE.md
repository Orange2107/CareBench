# Baseline Bayesian 搜索功能对齐清单（对齐 CrossVPT 脚本）

本文总结 `bayesian_search_crossvpt_hf.sh` 当前已实现的核心能力，供其他 baseline 搜索脚本对齐复用。

## A. 数据集与 CXR Encoder 对齐约束（当前统一设置）

- 任务固定为 `phenotype`，类别数固定为 `9`（由原 25 类筛选）。
- 9 个疾病标签固定为：  
  `CHF`、`CorAth/HD`、`Liver Dis.`、`COPD`、`HTN`、`AMI`、`CD`、`A. CVD`、`GIB`。  
  对应 25 类中的索引：`[8, 9, 17, 5, 13, 2, 7, 1, 15]`。
- 搜索脚本应固定传入：
  - `--task phenotype`
  - `--num_classes 9`
  - `--use_phenotype9 true`（或等价配置项，确保生效）
- CXR 编码器统一固定为 Hugging Face CheXpert ViT：
  - `--cxr_encoder hf_chexpert_vit`
  - `hf_model_id=codewithdark/vit-chest-xray`
- 以上数据与编码器设置属于"公平对齐条件"，不应放入超参数搜索空间。

## 1. 多 Seed 并行与多 GPU 调度

- 支持 `SEEDS=42,123,1234` 多 seed 搜索。
- 支持 `GPU=0,1,2` 多 GPU，并采用 round-robin 分配 seed 到 GPU。
- 每个 seed 子实验单独输出日志：
  - `<results_dir>/<exp_name>_seed<seed>/output.log`
- 迭代结果按 seed 聚合，计算 `mean/std`（而非单 seed）。

## 2. 每次迭代后自动清理 checkpoint

- 以主指标（当前为 `PRAUC_mean`）判断是否为全局最佳迭代。
- 若当前迭代不是最佳：立即删除该迭代所有 seed 对应训练目录下的 `checkpoints/`。
- 若当前迭代成为新最佳：删除历史非最佳迭代的 `checkpoints/`。
- 目标：全程仅保留"当前全局最佳"对应的 checkpoint，节省磁盘。

## 3. 最终收敛后再做一次全局清理

- `cleanup_final_checkpoints()` 在搜索结束后执行最终清理。
- 保证最终磁盘状态只保留最佳 iteration 的 checkpoint。

## 4. BEST 实验导出（保留三 seed）

- `export_best_experiment()` 会把最佳超参数对应的真实训练目录导出到稳定目录：
  - `BEST_<model>_<task>_fold...`
- 导出时保留每个 seed 的完整目录（`seed42/seed123/seed1234`），并写入 `best_metadata.json`。

## 5. 最佳迭代详细指标 CSV（跨 seed 聚合）

- 生成：`best_iteration_detailed_metrics.csv`
- 数据来源：每个 seed 真实训练目录下的 `test_set_results.yaml`
- 每个指标输出：`mean`、`std`、`formatted(mean ± std)`、`seeds_count`
- 已修复：不再依赖脆弱的目录名模糊匹配，改为"由 seed 日志反查真实训练目录"，并带回退匹配，避免只读到一个 seed。

## 6. Resume / Cleanup-only 支持

- 支持从历史 `results_summary.csv` 或优化器 checkpoint 恢复。
- 支持 `CLEANUP_ONLY=true`：不跑新实验，只按最佳迭代清理并导出统计。

## 7. 结果文件规范（建议所有 baseline 对齐）

每个 baseline 搜索目录建议至少包含：

- `results_summary.csv`：每次迭代聚合结果（含 `PRAUC_mean/std` 与 `is_best`）
- `best_params.txt`：最佳超参数摘要
- `best_iteration_detailed_metrics.csv`：最佳迭代跨 seed 详细指标
- `BEST_*` 目录：最佳设置下每个 seed 的完整导出结果

## 8. 其他 baseline 最小改造步骤

1. 引入与 CrossVPT 一致的多 seed 并行执行框架（ThreadPool + 多 GPU 分配）。
2. 在单次迭代结束后加入"是否最佳"判断与 checkpoint 删除逻辑。
3. 在搜索结束后增加最终清理函数，确保只保留最佳 checkpoint。
4. 增加 BEST 导出函数，按 seed 导出真实训练目录。
5. 增加 `best_iteration_detailed_metrics.csv` 生成逻辑，强制按多 seed 聚合。
6. 增加 `CLEANUP_ONLY` 模式，便于对既有搜索结果做后处理。

---

如果后续要批量对齐 `drfuse / shaspec / smil / flexmoe / healnet / medfuse`，可直接以 `bayesian_search_crossvpt_hf.sh` 为模板迁移以上模块，再替换各模型"独有超参数搜索空间"。

## 9. DrFuse 改进补充（2026-03-09 更新）

基于 HealNet 标准，DrFuse 脚本已完成以下 P0/P1 优先级改进，其他 baseline 可参考：

### 9.1 P0 改进：实验命名简化

**问题**：原命名包含 11 个固定参数，名称超长（~150 字符）

**修改前**：
```
bayes_iter1_fold1_lr0.000100_bs16_ehr_transformer_cxr_hf_chexpert_vit_hs256...
```

**修改后**（只包含 7 个搜索的 lambda 参数，~80 字符）：
```
drfuse_iter1_fold1_lds0.50_lde1.00_ldc0.30_lpe0.80_lpc1.20_lps0.60_laa0.90
```

**参数缩写对照**：
- `lds` = lambda_disentangle_shared
- `lde` = lambda_disentangle_ehr
- `ldc` = lambda_disentangle_cxr
- `lpe` = lambda_pred_ehr
- `lpc` = lambda_pred_cxr
- `lps` = lambda_pred_shared
- `laa` = lambda_attn_aux

### 9.2 P0 改进：已运行参数扫描机制

**功能**：避免重复运行相同的参数组合

**实现**：从 `results_summary.csv` 中提取已运行参数到 `self.already_run_params` 集合

**效果**：
- 中断恢复时自动跳过已运行参数
- 节省计算资源和时间
- 支持多次启动累积结果

### 9.3 P0 改进：CLEANUP_ONLY 模式增强

**使用方式**：
```bash
# 正常运行
./bayesian_search_drfuse_huggingvit.sh

# 清理模式（不运行新实验）
CLEANUP_ONLY=true ./bayesian_search_drfuse_huggingvit.sh
```

**实现逻辑**：在 `run_optimization()` 开头检查 `cleanup_only` 标志，如果为 true 则直接调用清理函数并返回。

### 9.4 P1 改进：详细指标保存函数

**生成文件**：`best_iteration_detailed_metrics.csv`

**内容格式**：
```csv
metric,mean,std,formatted,seeds_count
overall/PRAUC,0.6800,0.0123,0.6800 ± 0.0123,3
overall/ROC_AUC,0.7200,0.0089,0.7200 ± 0.0089,3
```

**关键实现**：
- 从每个种子的 `test_set_results.yaml` 读取
- 计算 mean ± std
- 支持所有指标（不仅限于 PRAUC）

### 9.5 P1 改进：增强的清理日志

**修改前**（简单 1 行）：
```
Final checkpoint cleanup done. Deleted 5 checkpoint dirs; kept best iteration 3.
```

**修改后**（详细多行，包含分隔线、最佳指标、参数、空间估算）：
```
============================================================
🧹 FINAL CHECKPOINT CLEANUP
============================================================
✅ Keeping only best iteration: 3
   Best PRAUC: 0.6800
   Best params: {'lds': 0.50, 'lde': 1.00, ...}

💾 CLEANUP SUMMARY
   Deleted 10 non-best iteration(s)
   Kept 3 best iteration(s)
   Freed ~15000MB disk space
============================================================
```

### 9.6 P1 改进：BEST 实验导出增强

**导出目录命名**：
```
BEST_drfuse_phenotype_fold1_lds0.50_lde1.00_ldc0.30_lpe0.80_lpc1.20_lps0.60_laa0.90_prauc0.6800/
├── seed42/
├── seed123/
├── seed1234/
└── best_metadata.json
```

**元数据**：包含最佳迭代信息、所有种子的导出路径、最佳参数、所有种子的指标等。

### 9.7 其他 baseline 对齐建议

**步骤 1**：在 `ver_name.py` 中定义搜索参数缩写格式

**步骤 2**：实现 `scan_existing_experiments()` 函数

**步骤 3**：简化实验命名，只包含搜索参数

**步骤 4**：添加 `CLEANUP_ONLY` 模式支持

## 9. 智能跳过已完成实验（2026-03-12 新增）✨

### 9.1 功能说明

贝叶斯搜索脚本现在支持**自动检测并跳过已完成的超参数组合**，避免重复运行实验。

**核心机制**：
1. 启动时扫描所有实验目录
2. 检查每个实验的 `test_set_results.yaml` 文件是否存在
3. 如果文件存在，认为该超参数组合已完成
4. 在贝叶斯优化时跳过这些组合

### 9.2 使用方法

**无需额外参数**，脚本会自动检测：

```bash
# 直接运行脚本，会自动跳过已完成的实验
./bayesian_search_healnet_hf.sh

# 脚本会自动：
# 1. 扫描 lightning_logs 目录下的所有实验
# 2. 检查每个实验的 test_set_results.yaml
# 3. 跳过已完成的超参数组合
# 4. 只运行新的组合
```

### 9.3 检测逻辑

```python
# 扫描父目录（lightning_logs）
parent_dir = os.path.dirname(self.results_dir)

# 匹配实验目录名：healnet_fold1_d2_fb1_mf10.0_seed42
exp_pattern = re.compile(r'healnet_fold\d+_d(\d+)_fb(\d+)_mf([\d.]+)_seed(\d+)')

for entry in os.listdir(parent_dir):
    match = exp_pattern.match(entry)
    if match:
        # 提取参数
        depth = int(match.group(1))
        num_freq_bands = int(match.group(2))
        max_freq = float(match.group(3))
        
        # 检查 test_set_results.yaml 是否存在
        results_yaml = os.path.join(entry_path, 'test_set_results.yaml')
        if os.path.exists(results_yaml):
            # 文件存在，标记为已完成
            self.already_run_params.add((depth, num_freq_bands, max_freq))
```

### 9.4 日志输出示例

```log
[2026-03-12 16:30:00] 🔍 Scanning for existing experiments...
[2026-03-12 16:30:00] 🔍 Scanning experiment directories for completed runs...
[2026-03-12 16:30:00]    Scanning directory: /hdd/bayesian_search_experiments/healnet/phenotype/lightning_logs
[2026-03-12 16:30:01]    ✅ Found completed: d1_fb1_mf5.0_seed42
[2026-03-12 16:30:01]    ✅ Found completed: d1_fb1_mf5.0_seed123
[2026-03-12 16:30:01]    ✅ Found completed: d1_fb1_mf5.0_seed1234
[2026-03-12 16:30:01]    ✅ Found completed: d2_fb1_mf10.0_seed42
[2026-03-12 16:30:01]    ⏳ Incomplete (no test_set_results.yaml): d3_fb4_mf10.0_seed42
[2026-03-12 16:30:01] ✅ Found 21 completed experiment(s) by checking test_set_results.yaml
[2026-03-12 16:30:01] 📊 Total unique parameter combinations to skip: 8

[2026-03-12 16:30:02] Starting Bayesian iteration 1: healnet_fold1_d3_fb4_mf10.0
[2026-03-12 16:30:02] ⚡ Running 3 seeds in parallel across 3 GPUs...

[2026-03-12 17:00:00] ⏭️  SKIP iteration 2: healnet_fold1_d2_fb2_mf5.0 (already completed)
[2026-03-12 17:00:00]    📊 Existing result: PRAUC_mean = 0.8234
```

### 9.5 适用场景

1. **中断恢复**：脚本中断后重新运行，自动跳过已完成的实验
2. **累积搜索**：多次启动脚本，累积更多超参数组合的搜索结果
3. **手动补充**：手动运行某些特定组合后，脚本会自动识别并跳过
4. **错误恢复**：某些实验失败后，修复问题重新运行，跳过成功的实验

### 9.6 注意事项

1. **不要手动删除 `test_set_results.yaml`**：
   - 该文件是实验完成的标志
   - 删除后脚本会认为实验未完成，重新运行

2. **确保 YAML 文件完整**：
   - 脚本只检查文件是否存在，不验证内容
   - 如果文件损坏或为空，可能导致跳过无效实验

3. **参数格式必须一致**：
   - 目录名必须匹配正则表达式
   - 例如：`healnet_fold1_d2_fb1_mf10.0_seed42`
   - 如果手动修改目录名，可能导致检测失败

### 9.7 与其他功能的配合

1. **与 Resume 功能配合**：
   - 即使没有贝叶斯优化检查点，也能跳过已完成实验
   - 双重保险：检查点 + 文件扫描

2. **与 CLEANUP_ONLY 模式配合**：
   - 清理模式下也会扫描已完成的实验
   - 只清理最佳迭代的 checkpoint

3. **与多 Seed 并行配合**：
   - 只要有一个 seed 完成，就跳过整个超参数组合
   - 避免部分重复运行

### 9.8 扩展到其他 baseline

其他模型的搜索脚本只需修改正则表达式即可复用：


**修改位置**：在 `scan_existing_experiments()` 方法中替换正则表达式。

---

## 10. 各模型搜索参数对照表

| 模型 | 搜索参数数量 | 参数类型 | 命名缩写示例（完整格式） |
|------|-------------|----------|-------------------------|
| **HealNet** | 3 | 架构参数 | `{model}_fold{fold}_d{depth}_fb{num_freq_bands}_mf{max_freq}_seed{seeds}` |
| **DrFuse** | 7 | Loss 权重 | `{model}_fold{fold}_lds{..:.4f}_lde{..:.4f}_ldc{..:.4f}_lpe{..:.4f}_lpc{..:.4f}_lps{..:.4f}_laa{..:.4f}_seed{seeds}` |
| **FlexMoE** | 4 | MoE 架构 | `{model}_fold{fold}_ne{num_experts}_nr{num_routers}_tk{top_k}_glw{gate_loss_weight:.4f}_seed{seeds}` |
| **ShaSpec** | 2 | Loss 权重 | `{model}_fold{fold}_a{alpha:.4f}_b{beta:.4f}_seed{seeds}` |
| **SMIL** | 6 | 元学习参数 | `{model}_fold{fold}_il{inner_loop}_mc{mc_size}_lri{lr_inner:.4f}_a{alpha:.4f}_b{beta:.4f}_t{temperature:.4f}_seed{seeds}` |
| **CrossVPT** | 3 | Prompt 参数 | `{model}_fold{fold}_seed{seed}_npt{num_prompt_tokens}_ptd{prompt_token_dropout}_pns{prompt_noise_std}` |
| **MedFuse** | 2 | 架构参数 | `{model}_fold{fold}_seed{seed}_ft{fusion_type}_ce{cxr_encoder}` |

**示例输出**：
```
# HealNet
healnet_fold1_d3_fb1_mf10.0_seed42

# DrFuse
drfuse_fold1_lds0.5000_lde1.0000_ldc0.3000_lpe0.8000_lpc1.2000_lps0.6000_laa0.9000_seed42

# FlexMoE
flexmoe_fold1_ne8_nr2_tk4_glw0.0100_seed42

# ShaSpec
shaspec_fold1_a0.0500_b0.1000_seed42

# SMIL
smil_fold1_il2_mc20_lri0.0005_a0.1000_b0.1500_t2.0000_seed42
```

**命名原则**（已统一）：
1. **基础格式**：`{model}_fold{fold}_<搜索参数>_seed{seeds}`
2. **只包含搜索的超参数**：固定参数（lr、batch_size 等）不显示
3. **使用简洁的缩写**：1-3 个字母（d=depth, fb=freqbands, mf=maxfreq）
4. **连续值格式**：使用 `:.4f` 保留 4 位小数（lambda 权重等）
5. **离散值格式**：直接显示整数值（depth、num_experts 等）
6. **元信息**：包含 fold 和 seed，便于追踪实验配置

---

**更新时间**：2026-03-12  
**更新内容**：
- 2026-03-12：新增智能跳过已完成实验功能（第 9 节）
- 2026-03-09：补充 DrFuse 脚本的 P0/P1 改进要点
