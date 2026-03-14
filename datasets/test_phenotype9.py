"""
Test script to verify Phenotype-9 filtering works correctly
"""
import pandas as pd
from pathlib import Path

try:
    from datasets.phenotype9 import (
        PHENOTYPE9_ABBREVIATIONS,
        PHENOTYPE9_LABELS,
        select_phenotype9_labels,
    )
except ImportError:
    from phenotype9 import (
        PHENOTYPE9_ABBREVIATIONS,
        PHENOTYPE9_LABELS,
        select_phenotype9_labels,
    )

# 测试数据路径
data_root = Path('/hdd/benchmark/benchmark_dataset/DataProcessing/benchmark_data/250827')

# 读取任意一个 split 文件
csv_file = data_root / 'splits' / 'fold1' / 'stays_train.csv'
df = pd.read_csv(csv_file)

print("=" * 80)
print("Testing Phenotype-9 Selection")
print("=" * 80)

# 获取所有 25 个 phenotype columns
all_phenotype_cols = df.columns[-26:-1].tolist()

print(f"\nAll {len(all_phenotype_cols)} phenotype labels:")
for i, col in enumerate(all_phenotype_cols, 1):
    print(f"  {i:2d}. {col}")

# 使用预定义的 9 个 phenotype labels
selected_classes, phenotype_9_indices = select_phenotype9_labels(all_phenotype_cols)

print(f"\n{'=' * 80}")
print(f"Phenotype-9 Selection (indices {phenotype_9_indices}):")
print(f"{'=' * 80}")
for i, cls in enumerate(selected_classes, 1):
    print(f"  {i}. {cls} ({PHENOTYPE9_ABBREVIATIONS[i-1]})")

print(f"\nTotal: {len(selected_classes)} labels (should be 9)")

# 验证包含 GIB 且不包含 PNA
gib_name = "Gastrointestinal hemorrhage"
if gib_name in selected_classes:
    print(f"\n✓ Verified: GIB is included")
else:
    print(f"\n❌ ERROR: GIB is missing!")

pna_name = "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)"
if pna_name in selected_classes:
    print("❌ ERROR: PNA should not be included!")
else:
    print("✓ Verified: PNA is not included")

# 验证所有期望的 labels 都在
expected_labels = PHENOTYPE9_LABELS

missing = set(expected_labels) - set(selected_classes)
if missing:
    print(f"❌ ERROR: Missing labels: {missing}")
else:
    print(f"✓ Verified: All 9 expected labels are present")

unexpected = set(selected_classes) - set(expected_labels)
if unexpected:
    print(f"❌ ERROR: Unexpected labels: {unexpected}")
else:
    print(f"✓ Verified: No unexpected labels are present")

print(f"\n{'=' * 80}")
print("Test completed!")
print(f"{'=' * 80}\n")
