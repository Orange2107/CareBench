# Phenotype-9: 9 种心血管/呼吸系统疾病多标签分类

## 概述

Phenotype-9 是从原始 25 个 phenotype labels 中精选的 9 个心血管和呼吸系统相关疾病标签，用于多标签分类任务。

## 选择的 9 个 Labels

| 索引 | 疾病名称 | 缩写 | 类别 |
|------|---------|------|------|
| 8 | Congestive heart failure; nonhypertensive | CHF | 心血管疾病 |
| 9 | Coronary atherosclerosis and other heart disease | CorAth/HD | 心血管疾病 |
| 17 | Other liver diseases | Liver Dis. | 肝脏疾病 |
| 5 | Chronic obstructive pulmonary disease and bronchiectasis | COPD | 呼吸系统疾病 |
| 13 | Essential hypertension | HTN | 心血管疾病 |
| 2 | Acute myocardial infarction | AMI | 心血管疾病 |
| 7 | Conduction disorders | CD | 心血管疾病 |
| 1 | Acute cerebrovascular disease | A. CVD | 脑血管疾病 |
| 15 | Gastrointestinal hemorrhage | GIB | 消化系统疾病 |

缩写补全：
- `CHF`: Congestive heart failure; nonhypertensive
- `CorAth/HD`: Coronary atherosclerosis and other heart disease
- `Liver Dis.`: Other liver diseases
- `COPD`: Chronic obstructive pulmonary disease and bronchiectasis
- `HTN`: Essential hypertension
- `AMI`: Acute myocardial infarction
- `CD`: Conduction disorders
- `A. CVD`: Acute cerebrovascular disease
- `GIB`: Gastrointestinal hemorrhage

## 使用方法

### 方法 1: 使用配置文件（推荐）

```bash
python main.py --train_config configs/train_configs/crossvpt_hf_phenotype.yaml
```

### 方法 2: 命令行参数

```bash
python main.py --model crossvpt --task phenotype --use_phenotype9 --fold 1
```

### 方法 3: 代码中使用

```python
from datasets.dataset import create_data_loaders

# 使用 Phenotype-9
train_loader, val_loader, test_loader = create_data_loaders(
    ehr_data_dir='/path/to/data',
    task='phenotype',
    use_phenotype9=True,  # 启用 Phenotype-9
    batch_size=16,
    # ... 其他参数
)

# 使用全部 25 个 labels（默认行为）
train_loader, val_loader, test_loader = create_data_loaders(
    ehr_data_dir='/path/to/data',
    task='phenotype',
    use_phenotype9=False,  # 或不指定该参数
    batch_size=16,
)
```

## 配置参数

### `use_phenotype9` (bool)

- **默认值**: `False`
- **说明**: 是否使用 Phenotype-9 筛选
- **效果**:
  - `True`: 使用 9 个精选的 phenotype labels
  - `False`: 使用全部 25 个 phenotype labels（原始行为）

## 输出示例

当启用 `use_phenotype9=True` 时，训练开始时会看到以下输出：

```
Using Phenotype-9 (indices [8, 9, 17, 5, 13, 2, 7, 1, 15]):
  1. Congestive heart failure; nonhypertensive
  2. Coronary atherosclerosis and other heart disease
  3. Other liver diseases
  4. Chronic obstructive pulmonary disease and bronchiectasis
  5. Essential hypertension
  6. Acute myocardial infarction
  7. Conduction disorders
  8. Acute cerebrovascular disease
  9. Gastrointestinal hemorrhage
```

## 标签权重计算

Phenotype-9 任务会自动计算每个 label 的权重以处理类别不平衡问题。支持以下方法：

- `balanced`: 使用 sklearn 的 balanced 权重
- `inverse`: 正类样本数的倒数
- `sqrt_inverse`: 正类样本数倒数的平方根
- `log_inverse`: 正类样本数倒数的对数

```bash
python main.py --model crossvpt --task phenotype --use_phenotype9 \
  --use_label_weights --label_weight_method balanced
```

## 技术实现

### 索引映射

Phenotype-9 使用固定的标签名进行选择，再映射回当前 CSV 中的索引位置：

```python
PHENOTYPE9_LABELS = [
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Other liver diseases",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Essential hypertension",
    "Acute myocardial infarction",
    "Conduction disorders",
    "Acute cerebrovascular disease",
    "Gastrointestinal hemorrhage",
]
```

这样可以避免因为列顺序变化而误选错误标签。

### 向后兼容性

- 默认行为保持不变（`use_phenotype9=False`）
- 所有现有代码和配置文件无需修改即可继续工作
- 只在明确启用 `use_phenotype9` 时才应用筛选

## 验证测试

运行测试脚本验证 Phenotype-9 选择是否正确：

```bash
python datasets/test_phenotype9.py
```

## 参考文献

这些 phenotype labels 基于 Elixhauser 共病指数，广泛用于医疗数据分析中的疾病分类和风险评估。
