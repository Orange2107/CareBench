import pandas as pd

try:
    from datasets.phenotype9 import select_phenotype9_labels
except ImportError:
    from phenotype9 import select_phenotype9_labels

# 查看 fold1 的 train/val/test split 文件
fold = 1  # 或其他 fold (1-5)
partition = 'train'  # 或 'val', 'test'

# 路径格式：{data_root}/splits/fold{fold}/stays_{partition}.csv
csv_path = f'/hdd/benchmark/benchmark_dataset/DataProcessing/benchmark_data/250827/splits/fold{fold}/stays_{partition}.csv'

df = pd.read_csv(csv_path)
phenotype_cols = df.columns[-26:-1].tolist()

print("Phenotype labels (25 个):")
for i, col in enumerate(phenotype_cols, 1):
    print(f"{i}. {col}")


selected_labels, phenotype_9_indices = select_phenotype9_labels(phenotype_cols)

print("Selected phenotype labels:")
for i, (idx, label) in enumerate(zip(phenotype_9_indices, selected_labels), 1):
    print(f"{i}. Index {idx}: {label}")
