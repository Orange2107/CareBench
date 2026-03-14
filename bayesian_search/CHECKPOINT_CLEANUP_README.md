# 智能 Checkpoint 清理机制

## 📋 概述

Bayesian 超参数搜索会自动清理非最佳的 checkpoints，以节省磁盘空间。

## 🎯 问题背景

- **原始问题**：20 次迭代 × 3 个 seeds = 60 个 checkpoints
- **空间占用**：每个 checkpoint ~500MB，总计约 **30GB**
- **浪费**：大部分中间结果的 checkpoints 最终不会被使用

## ✨ 解决方案

实现了**两级智能清理机制**：

### 1️⃣ 实时清理（每次迭代后）

在每次迭代完成后，立即判断是否保留 checkpoints：

**保留条件**（满足任一即可）：
- ✅ **当前迭代是新的最佳**（PRAUC 最高）
- ✅ **当前 PRAUC 排名前 3**
- ⚠️ **最后 3 次迭代**（用于调试，在最终清理时处理）

**清理逻辑**：
```python
if 不是最佳 and 不在前 3 and 不是最后 3 次迭代:
    删除当前迭代的所有 checkpoints
```

**优势**：
- 🚀 即时释放空间
- 📊 保持磁盘使用稳定
- 🔍 保留有价值的中间结果

### 2️⃣ 最终清理（优化完成后）

在贝叶斯优化完全结束后，进行最终清理：

**保留策略**：
1. **最佳迭代**（PRAUC 最高）
2. **最后 3 次迭代**（便于调试最近的搜索）
3. **Top-3 PRAUC 迭代**（表现最好的 3 个）

**清理效果**：
```
原始：60 个 checkpoints × 500MB = 30GB
清理后：~9 个 checkpoints × 500MB = 4.5GB
节省：~25.5GB (85% 空间)
```

## 📊 判断指标

### 主指标：`overall/PRAUC`

从训练日志中提取 `overall/PRAUC` 作为评判标准：

```python
# 从日志中提取 PRAUC
patterns = {
    'PRAUC': r"overall/PRAUC:\s*([0-9]+\.[0-9]+)"
}

# 比较当前与历史最佳
if current_prauc > best_prauc:
    保留 checkpoints
    更新最佳记录
else:
    可能删除（取决于其他条件）
```

## 🔧 配置选项

在 `bayesian_search_crossvpt_hf.sh` 中调整清理策略：

```bash
# 当前配置（推荐）
# - 实时清理：删除非最佳且非 top-3 的 checkpoints
# - 最终清理：保留最佳 + 最后 3 次 + top-3

# 如果想保留更多 checkpoints：
# 修改 cleanup_final_checkpoints() 中的：
last_n_iterations = 5  # 保留最后 5 次（默认 3）
top_n_prauc = 5        # 保留 top-5（默认 3）

# 如果想完全禁用清理：
# 注释掉 run_optimization() 中的：
# self.cleanup_final_checkpoints()
```

## 📈 清理流程示例

### 迭代 1-5（初始随机探索）
```
Iter 1: PRAUC=0.65 → 保留（首个结果）
Iter 2: PRAUC=0.68 → 保留（新的最佳 ✨）
Iter 3: PRAUC=0.66 → 保留（top-3）
Iter 4: PRAUC=0.64 → 删除 ❌（不是最佳，不在 top-3）
Iter 5: PRAUC=0.67 → 保留（top-3）
```

### 迭代 15（接近收敛）
```
当前最佳：0.75（Iter 12）
Iter 15: PRAUC=0.72 → 删除 ❌
释放空间：3 个 checkpoints × 500MB = 1.5GB
```

### 最终清理
```
保留：
✅ Iter 20 (PRAUC=0.78, 最佳)
✅ Iter 18 (PRAUC=0.77, top-3)
✅ Iter 12 (PRAUC=0.75, top-3)
✅ Iter 19 (最后 3 次)
✅ Iter 20 (最后 3 次)

删除：
❌ 其他所有 iterations
```

## 💾 空间节省估算

| 场景 | 原始空间 | 清理后 | 节省 |
|------|---------|--------|------|
| **20 次迭代** | 30GB | 4.5GB | **25.5GB (85%)** |
| **30 次迭代** | 45GB | 6.0GB | **39GB (87%)** |
| **50 次迭代** | 75GB | 7.5GB | **67.5GB (90%)** |

## ⚠️ 注意事项

### 1. 清理时机
- ✅ **实时清理**：每次迭代完成后立即执行
- ✅ **最终清理**：所有迭代完成后执行

### 2. 安全性
- 🔒 **只删除 checkpoints**，不删除日志文件
- 🔒 **保留所有实验记录**（output.log, results_summary.csv）
- 🔒 **保留最佳模型**用于后续测试

### 3. 恢复能力
- 如果误删，可以从 `results_summary.csv` 找到最佳迭代
- 使用最佳参数重新训练即可

### 4. 调试模式
如果想保留所有 checkpoints 用于调试：

```bash
# 临时禁用清理
# 注释掉这两行：
# self.cleanup_checkpoints_if_needed()  # 在 run_experiment_with_seeds 中
# self.cleanup_final_checkpoints()      # 在 run_optimization 中
```

## 📝 日志输出示例

```
[2026-03-05 10:30:00] Starting Bayesian iteration 15: bayes_iter15_...
[2026-03-05 11:45:00] Iteration 15 - PRAUC: 0.7234±0.0012
[2026-03-05 11:45:01] 🗑️ Cleaning up checkpoints for iteration 15 (PRAUC=0.7234, Best=0.7523)
[2026-03-05 11:45:02]    Deleted: .../bayes_iter15_fold1_.../checkpoints
[2026-03-05 11:45:03]    Deleted: .../bayes_iter15_fold1_.../checkpoints
[2026-03-05 11:45:04]    Deleted: .../bayes_iter15_fold1_.../checkpoints
[2026-03-05 11:45:05] 💾 Freed ~1500MB disk space

============================================================
🧹 FINAL CHECKPOINT CLEANUP
============================================================
✅ Keeping best iteration: 20 (PRAUC=0.7823)
✅ Keeping last 3 iterations: [18, 19, 20]
✅ Keeping top-3 PRAUC iterations: [20, 18, 12]

📊 Total iterations to keep: 5 / 20

💾 CLEANUP SUMMARY
   Deleted 45 checkpoint directories
   Freed ~22500MB (21.97GB) disk space
   Kept 5 iterations' checkpoints
============================================================
```

## 🎯 最佳实践

1. **使用默认配置**：适合大多数研究场景
2. **监控日志**：查看哪些 checkpoints 被保留/删除
3. **最终验证**：清理后检查保留了哪些 iterations
4. **调整策略**：根据磁盘空间和研究需求调整保留数量

## 📚 相关文件

- `bayesian_search_crossvpt_hf.sh`: 主脚本（包含清理逻辑）
- `results_summary.csv`: 所有迭代的结果汇总
- `best_params.txt`: 最佳参数配置
- `bayesian_optimization_result.pkl`: 完整的贝叶斯优化对象

## 🆘 故障排除

### Q: 如果清理后需要某个被删除的 checkpoint 怎么办？
A: 使用 `best_params.txt` 中的参数重新训练该配置

### Q: 清理功能是否会影响贝叶斯优化过程？
A: 不会！清理只删除 checkpoint 文件，不影响优化算法

### Q: 如何确认清理是否正常工作？
A: 查看日志中的 `🗑️` 和 `✅` 标记，确认清理行为
