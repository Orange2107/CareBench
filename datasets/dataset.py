import os
import pickle
from pathlib import Path
import ast
import re

import yaml
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import random

import torch
import lightning as L
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight


# set dataloader seed
def set_seed(_seed):
    global seed
    seed = _seed
    print(f"set the seed of dataloader as :{seed}")
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If multiple GPUs


class MultiModalMIMIC(Dataset):
    def __init__(self, seed, data_root, fold, partition, task,
                 time_limit=48, normalization='robust_scale', ehr_time_step=1,
                 matched_subset=True, imagenet_normalization=True,
                 preload_images=False, pkl_dir=None, attribution_cols=None, one_hot=None, use_triplet=False,
                 resized_base_path='/research/mimic_cxr_resized',
                 image_meta_path="/hdd/datasets/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv",
                 use_demographics=False, demographic_cols=None,
                 use_label_weights=False, label_weight_method='balanced', custom_label_weights=None,
                 cxr_dropout_rate=0.0, cxr_dropout_seed=None, demographics_in_model_input=False):
        self.seed = seed
        # Save initial random state
        self.random_state = random.getstate()
        self.np_random_state = np.random.get_state()
        self.torch_random_state = torch.get_rng_state()
        
        self.task = task
        self.normalization = normalization
        self.ehr_time_step = ehr_time_step
        self.time_limit = time_limit
        self.matched_subset = matched_subset
        self.one_hot = one_hot
        self.preload_images = {}
        self.resized_base_path = resized_base_path
        self.use_triplet = use_triplet
        
        # Label weight configuration
        self.use_label_weights = use_label_weights
        self.label_weight_method = label_weight_method
        self.custom_label_weights = custom_label_weights
        
        # CXR dropout configuration for robustness evaluation
        self.cxr_dropout_rate = cxr_dropout_rate
        self.cxr_dropout_seed = cxr_dropout_seed if cxr_dropout_seed is not None else seed
        self.partition = partition  # Store partition for dropout logic
        
        # Initialize empty dropout samples set (will be populated after data loading)
        self.dropped_cxr_samples = set()
        
        # Demographic features configuration
        self.use_demographics = use_demographics
        if demographic_cols is None:
            self.demographic_cols = ['age', 'gender', 'admission_type', 'race']
        else:
            self.demographic_cols = demographic_cols

        # Define ethnicity mapping to 5 major categories
        self.ethnicity_mapping = {
            'WHITE': 'WHITE',
            'WHITE - OTHER EUROPEAN': 'WHITE',
            'WHITE - RUSSIAN': 'WHITE',
            'WHITE - EASTERN EUROPEAN': 'WHITE',
            'WHITE - BRAZILIAN': 'WHITE',
            'BLACK/AFRICAN AMERICAN': 'BLACK/AFRICAN AMERICAN',
            'BLACK/CAPE VERDEAN': 'BLACK/AFRICAN AMERICAN',
            'BLACK/CARIBBEAN ISLAND': 'BLACK/AFRICAN AMERICAN',
            'BLACK/AFRICAN': 'BLACK/AFRICAN AMERICAN',
            'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC/LATINO',
            'HISPANIC OR LATINO': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - DOMINICAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - SALVADORAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - MEXICAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - CUBAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - HONDURAN': 'HISPANIC/LATINO',
            'HISPANIC/LATINO - CENTRAL AMERICAN': 'HISPANIC/LATINO',
            'ASIAN': 'ASIAN',
            'ASIAN - CHINESE': 'ASIAN',
            'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
            'ASIAN - ASIAN INDIAN': 'ASIAN',
            'ASIAN - KOREAN': 'ASIAN',
            'OTHER': 'OTHER',
            'AMERICAN INDIAN/ALASKA NATIVE': 'OTHER',
            'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'OTHER',
            'SOUTH AMERICAN': 'OTHER',
            'MULTIPLE RACE/ETHNICITY': 'OTHER',
            'UNKNOWN': 'OTHER',  # Merged with OTHER
            'UNABLE TO OBTAIN': 'OTHER',  # Merged with OTHER
            'PATIENT DECLINED TO ANSWER': 'OTHER'  # Merged with OTHER
        }

        # Gender mapping for consistency
        self.gender_mapping = {
            'M': 'MALE',
            'F': 'FEMALE'
        }

        print(f"In our data time_limit is {time_limit}")

        if attribution_cols is None:
            print(f"attribution_cols is None")
            self.attribution_cols = ['age', 'gender', 'admission_type', 'race']
        else:
            self.attribution_cols = attribution_cols

        self.data_root = Path(data_root)
        if pkl_dir is not None:
            demo_suffix = "_demo" if use_demographics else ""
            ehr_pkl_fpath = Path(pkl_dir) / f'{task}_fold{fold}_{partition}_timestep{ehr_time_step}_{normalization}_{"matched" if matched_subset else "full"}{demo_suffix}_ts.pkl'
        else:
            ehr_pkl_fpath = None

        # EHR data prepare------------
        # load EHR metadata and feature statistics (full data)
        meta_files_root = self.data_root/'splits'/f'fold{fold}'
        self.ehr_meta = pd.read_csv(meta_files_root/f'stays_{partition}.csv') 
        with open(meta_files_root/'train_stats.yaml', 'r') as f:
            self.train_stats = yaml.safe_load(f)

        # get ehr cols name
        with open(self.data_root/'splits'/'features.yaml', 'r') as f:
            features_yaml = yaml.safe_load(f)

        self.chartlab_feature = features_yaml['chartlab_feature']
        # self.treatment_feature = features_yaml['treatment_feature']
        # self.chartlab_feature = ['vitalsign_dbp',
        #                         'vitalsign_glucose',
        #                         'vitalsign_heart_rate',
        #                         'vitalsign_mbp',
        #                         'vitalsign_resp_rate',
        #                         'vitalsign_sbp',
        #                         'vitalsign_spo2',
        #                         'vitalsign_temperature',
        #                         'gcs_gcs',
        #                         'gcs_gcs_eyes',
        #                         'gcs_gcs_motor',
        #                         'gcs_gcs_verbal']
                                
        # self.features = self.chartlab_feature + self.treatment_feature
        self.features = self.chartlab_feature
        # 只为非 rhythm_ 特征生成 mask
        self.mask_name = [f"{feat}_mask" for feat in self.chartlab_feature if not feat.startswith('rhythm_')]

        print(f"now self.features is {self.features}")
        print(f"now self.mask_name is {self.mask_name}")

        # get attribute of each col
        self.features_stats = {
            stat: np.array([self.train_stats[feat][stat] for feat in self.chartlab_feature]).astype(float)
            for stat in ['iqr','max','mean','median','min','std']
        }
        self.features_no_normalization = [feat for feat in self.features if not self.train_stats[feat]['normalize']]


        #  set imputation value 
        self.default_imputation = {feat: self.train_stats[feat]['median'] for feat in self.features}

        # choice paired data
        print(f"In our dataset, match is {self.matched_subset}")
        print(f"before matched data length of data is {len(self.ehr_meta['stay_id'].tolist())}")
        if self.matched_subset:
            self.ehr_meta = self.ehr_meta[(self.ehr_meta['valid_cxrs'] != '[]') & (self.ehr_meta['valid_cxrs'].notna())]
            print(f"after matched length is {len(self.ehr_meta['stay_id'].tolist())}")

        # 首先处理 labels（包括数据过滤）
        if task == 'mortality':
            self.CLASSES = ['Mortality']
            self.targets = self.ehr_meta['icu_mortality'].values
        elif task == 'phenotype':
            self.CLASSES = self.ehr_meta.columns[-26:-1].tolist()
            self.targets = self.ehr_meta[self.CLASSES].values.astype(np.float32)
        elif task == 'los':
            # 定义 LoS 类别名称
            self.CLASSES = ["2-3d", "3-4d", "4-5d", "5-6d", "6-7d", "7-14d", "14+d"]
            
            # 过滤掉小于2天的数据
            print(f"Original data size: {len(self.ehr_meta)}")
            self.ehr_meta = self.ehr_meta[self.ehr_meta['los'] >= 2.0]
            print(f"After filtering LoS >= 2 days: {len(self.ehr_meta)}")
            
            # 重新定义分箱：从2天开始，去掉0和1类
            LOS_bins = [2, 3, 4, 5, 6, 7, 14, 102] 
            LOS_labels = [0, 1, 2, 3, 4, 5, 6]  # 重新编号为0-6，共7个类别
            
            print(f"Applying LoS binning to {len(self.ehr_meta)} samples...")
            print(f"Using 7 LoS bins (excluding 0-2 days):")
            bin_descriptions = [
                "[2, 3) days",   # 新的bin 0
                "[3, 4) days",   # 新的bin 1  
                "[4, 5) days",   # 新的bin 2
                "[5, 6) days",   # 新的bin 3
                "[6, 7) days",   # 新的bin 4
                "[7, 14) days",  # 新的bin 5
                "[14, 102] days" # 新的bin 6
            ]
            for i, desc in enumerate(bin_descriptions):
                print(f"  Bin {i}: {desc}")
            
            print(f"Original LoS stats (after filtering):")
            print(self.ehr_meta['los'].describe())
            
            # Apply binning to los column
            self.ehr_meta['los_bin'] = pd.cut(
                self.ehr_meta['los'],
                bins=LOS_bins,
                labels=LOS_labels,
                right=False,        # 左闭右开
                include_lowest=True
            ).astype('int8')
            
            # 重要：验证标签范围
            min_label = self.ehr_meta['los_bin'].min()
            max_label = self.ehr_meta['los_bin'].max()
            print(f"Label range after binning: [{min_label}, {max_label}]")
            
            # 检查是否有异常值
            if min_label < 0 or max_label >= 7:
                print(f"⚠️  WARNING: Labels outside expected range [0, 6]!")
                print(f"Min label: {min_label}, Max label: {max_label}")
                # 可选：截断异常值
                self.ehr_meta['los_bin'] = self.ehr_meta['los_bin'].clip(0, 6)
                print("Labels clipped to [0, 6] range")
            
            print(f"LoS binning applied. Class distribution:")
            los_counts = self.ehr_meta['los_bin'].value_counts().sort_index()
            for bin_idx, count in los_counts.items():
                percentage = count / len(self.ehr_meta) * 100
                print(f"  Bin {bin_idx} {bin_descriptions[bin_idx]}: {count} samples ({percentage:.2f}%)")
            
            # 存储分箱信息供后续使用
            self.los_bins = LOS_bins
            self.los_bin_descriptions = bin_descriptions
            self.targets = self.ehr_meta['los_bin'].values
            self.num_classes = 7
            print(f"LoS prediction configured with {self.num_classes} classes (2+ days)")
        else:
            raise ValueError(f'Unknown task `{task}`. Only mortality, phenotype, and los are supported')

        # 数据过滤后，更新 stay_ids
        self.stay_ids = self.ehr_meta['stay_id'].tolist()

        # 然后加载/保存 pkl（基于过滤后的数据）
        if ehr_pkl_fpath and ehr_pkl_fpath.exists():
            with open(ehr_pkl_fpath, 'rb') as f:
                self.normalized_data, self.missing_masks = pickle.load(f)
            print('Time series data loaded from pkl file.')
        else:
            self.normalized_data, self.missing_masks = self.load_and_normalize_time_series()
            # Define pkl file path for caching processed data
            demo_suffix = "_demo" if use_demographics else ""
            ehr_pkl_fpath = f'{task}_fold{fold}_{partition}_timestep{ehr_time_step}_{normalization}_{"matched" if matched_subset else "full"}{demo_suffix}_ts.pkl'
            if ehr_pkl_fpath:
                if not os.path.exists(os.path.join(self.data_root, 'data_pkls')):
                    os.makedirs(os.path.join(self.data_root, 'data_pkls'))
                with open(os.path.join(self.data_root, 'data_pkls', ehr_pkl_fpath), 'wb') as f:
                    pickle.dump([self.normalized_data, self.missing_masks], f)

        # Prepare demographic features if enabled
        if self.use_demographics:
            self.prepare_demographic_features()
        else:
            self.demo_feature_dim = 0
            self.demographic_data = {}

        # Calculate total input dimension
        self.base_ehr_dim = len(self.features) + len(self.mask_name)  # EHR features + masks
        self.input_dim = self.base_ehr_dim + (self.demo_feature_dim if getattr(self, 'demographics_in_model_input', False) else 0)
        
        print(f"Dataset input dimensions:")
        print(f"  Base EHR features: {len(self.features)}")
        print(f"  EHR masks: {len(self.mask_name)}")
        print(f"  Demographics: {self.demo_feature_dim}")
        print(f"  Total input dim: {self.input_dim}")

        # CXR data prepare-----------------------

        # load CXR metadata
        self.cxr_meta = pd.read_csv(image_meta_path)

        # define transformation for CXR
        cxr_transform = [transforms.Resize(256)]
        if partition == 'train':
            cxr_transform += [
                transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)),
            ]
        cxr_transform += [
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
        if imagenet_normalization:
            cxr_transform += [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.cxr_transform = transforms.Compose(cxr_transform)

        # Calculate label weights if enabled
        if self.use_label_weights:
            self.label_weights = self.calculate_label_weights()
            print(f"Computed label weights: {self.label_weights}")
        else:
            self.label_weights = None
        
        self.meta_attr = self.ehr_meta.set_index('stay_id')
        self.meta_attr = self.meta_attr[self.attribution_cols]
        
        # Initialize CXR dropout samples (one-time for entire dataset)
        if self.cxr_dropout_rate > 0.0 and self.partition in ['train', 'val']:
            cxr_dropout_rng = np.random.RandomState(self.cxr_dropout_seed)
            n_samples = len(self.stay_ids)
            n_drop = int(n_samples * self.cxr_dropout_rate)
            
            if n_drop > 0:
                drop_indices = cxr_dropout_rng.choice(n_samples, size=n_drop, replace=False)
                self.dropped_cxr_samples = set(drop_indices)
                print(f"CXR dropout initialized for {self.partition}: {n_drop}/{n_samples} samples will have CXR permanently dropped ({self.cxr_dropout_rate*100:.1f}%)")
            else:
                self.dropped_cxr_samples = set()
                print(f"CXR dropout rate too low for {self.partition}: no samples will be dropped")
        elif self.cxr_dropout_rate > 0.0:
            print(f"CXR dropout disabled for {self.partition} (only applied to train/val)")
        else:
            self.dropped_cxr_samples = set()

    def calculate_label_weights(self):
        """
        计算label weights用于处理类别不平衡问题
        
        Returns:
            torch.Tensor: 计算得到的label weights
        """
        print(f"\n=== Calculating Label Weights ===")
        print(f"Task: {self.task}")
        print(f"Method: {self.label_weight_method}")
        
        if self.task == 'mortality':
            return self._calculate_mortality_weights()
        elif self.task == 'phenotype':
            return self._calculate_phenotype_weights()
        elif self.task == 'los':
            return self._calculate_los_weights()
        else:
            raise ValueError(f"Unsupported task: {self.task}")
    
    def _calculate_mortality_weights(self):
        """
        计算mortality任务的label weights（二分类）
        """
        targets = self.targets
        
        # 确保targets是一维数组
        if targets.ndim > 1:
            targets = targets.flatten()
        
        # 统计类别分布
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        print(f"Mortality task class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} samples ({count/len(targets)*100:.2f}%)")
        
        if self.label_weight_method == 'custom' and self.custom_label_weights is not None:
            # 使用自定义权重
            weights = []
            for cls in unique_classes:
                cls_name = self.CLASSES[cls] if cls < len(self.CLASSES) else str(cls)
                weight = self.custom_label_weights.get(cls_name, 1.0)
                weights.append(weight)
            weights = np.array(weights)
        else:
            # 使用sklearn的compute_class_weight
            weights = compute_class_weight(
                class_weight=self.label_weight_method,
                classes=unique_classes,
                y=targets
            )
        
        # 创建完整的权重向量
        full_weights = np.zeros(len(unique_classes))
        for i, cls in enumerate(unique_classes):
            full_weights[cls] = weights[i]
        
        print(f"Computed weights: {full_weights}")
        return torch.tensor(full_weights, dtype=torch.float32)
    
    def _calculate_los_weights(self):
        """
        计算LoS任务的label weights（多分类）
        """
        targets = self.targets
        
        # 确保targets是一维数组
        if targets.ndim > 1:
            targets = targets.flatten()
        
        # 统计类别分布
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        print(f"LoS task class distribution:")
        
        # 使用存储的分箱描述
        for cls, count in zip(unique_classes, class_counts):
            if hasattr(self, 'los_bin_descriptions') and cls < len(self.los_bin_descriptions):
                bin_desc = self.los_bin_descriptions[cls]
            else:
                # 回退到默认描述（从2天开始的分箱）
                bin_descriptions = [
                    "[2, 3) days", "[3, 4) days", "[4, 5) days", "[5, 6) days",
                    "[6, 7) days", "[7, 14) days", "[14, 102] days"
                ]
                bin_desc = bin_descriptions[cls] if cls < len(bin_descriptions) else f"Class_{cls}"
            
            print(f"  Class {cls} {bin_desc}: {count} samples ({count/len(targets)*100:.2f}%)")
        
        if self.label_weight_method == 'custom' and self.custom_label_weights is not None:
            # 使用自定义权重
            weights = []
            for cls in unique_classes:
                weight = self.custom_label_weights.get(f"los_bin_{cls}", 1.0)
                weights.append(weight)
            weights = np.array(weights)
        else:
            # 使用sklearn的compute_class_weight
            weights = compute_class_weight(
                class_weight=self.label_weight_method,
                classes=unique_classes,
                y=targets
            )
        
        # 创建完整的权重向量（7个类别）
        full_weights = np.ones(7)  # 初始化为1，改为7个类别
        for i, cls in enumerate(unique_classes):
            if cls < 7:  # 确保索引在范围内
                full_weights[cls] = weights[i]
        
        print(f"Computed weights: {full_weights}")
        return torch.tensor(full_weights, dtype=torch.float32)

    def _calculate_phenotype_weights(self):
        """
        计算phenotype任务的label weights（多标签分类）
        """
        targets = self.targets
        
        if targets.ndim != 2:
            raise ValueError(f"Phenotype targets should be 2D array, got shape {targets.shape}")
        
        num_classes = targets.shape[1]
        print(f"Phenotype task class distribution:")
        weights = []
        
        for i in range(num_classes):
            class_targets = targets[:, i]
            positive_count = np.sum(class_targets == 1)
            negative_count = np.sum(class_targets == 0)
            total_count = len(class_targets)
            
            print(f"  {self.CLASSES[i]}:")
            print(f"    Positive: {positive_count} ({positive_count/total_count*100:.2f}%)")
            print(f"    Negative: {negative_count} ({negative_count/total_count*100:.2f}%)")
            
            if self.label_weight_method == 'custom' and self.custom_label_weights is not None:
                # 使用自定义权重
                weight = self.custom_label_weights.get(self.CLASSES[i], 1.0)
            else:
                # 计算权重
                if self.label_weight_method == 'balanced':
                    weight = compute_class_weight(
                        class_weight='balanced',
                        classes=np.array([0, 1]),  
                        y=class_targets
                    )[1]  # 取正类的权重
                elif self.label_weight_method == 'inverse':
                    # 使用样本数量的倒数
                    weight = total_count / (2 * positive_count) if positive_count > 0 else 1.0
                elif self.label_weight_method == 'sqrt_inverse':
                    # 使用样本数量倒数的平方根
                    weight = np.sqrt(total_count / (2 * positive_count)) if positive_count > 0 else 1.0
                elif self.label_weight_method == 'log_inverse':
                    # 使用样本数量倒数的对数
                    weight = np.log(total_count / (2 * positive_count) + 1) if positive_count > 0 else 1.0
                else:
                    weight = 1.0
            
            weights.append(weight)
            print(f"    Weight: {weight:.4f}")
        
        weights = np.array(weights)
        print(f"Overall weights: {weights}")
        return torch.tensor(weights, dtype=torch.float32)

    def get_label_weights(self):
        """
        获取计算好的label weights
        
        Returns:
            torch.Tensor: label weights，如果未启用则返回None
        """
        return self.label_weights

    def prepare_demographic_features(self):
        """Prepare and encode demographic features with proper mapping and one-hot encoding"""
        print(f"Preparing demographic features: {self.demographic_cols}")
        
        # Get demographic data from ehr_meta
        demo_df = self.ehr_meta.set_index('stay_id')[self.demographic_cols].copy()
        
        # Process each demographic feature
        categorical_features = []
        numerical_features = []
        
        for col in self.demographic_cols:
            if col == 'race':
                # Apply ethnicity mapping
                demo_df[col] = demo_df[col].map(self.ethnicity_mapping).fillna('OTHER')
                # Fill missing values with mode
                mode_value = demo_df[col].mode()[0] if not demo_df[col].mode().empty else 'OTHER'
                demo_df[col] = demo_df[col].fillna(mode_value)
                categorical_features.append(col)
                print(f"Race categories after mapping: {demo_df[col].unique()}")
                
            elif col == 'gender':
                # Apply gender mapping
                demo_df[col] = demo_df[col].map(self.gender_mapping)
                # Fill missing values with mode
                mode_value = demo_df[col].mode()[0] if not demo_df[col].mode().empty else 'MALE'
                demo_df[col] = demo_df[col].fillna(mode_value)
                categorical_features.append(col)
                print(f"Gender categories: {demo_df[col].unique()}")
                
            elif col == 'admission_type':
                # Fill missing values with mode
                mode_value = demo_df[col].mode()[0] if not demo_df[col].mode().empty else 'EMERGENCY'
                demo_df[col] = demo_df[col].fillna(mode_value)
                categorical_features.append(col)
                print(f"Admission type categories: {demo_df[col].unique()}")
                
            elif col == 'age':
                # Handle age as numerical feature
                # Fill missing values with median
                median_value = demo_df[col].median()
                demo_df[col] = demo_df[col].fillna(median_value)
                # Normalize age
                mean_age = demo_df[col].mean()
                std_age = demo_df[col].std()
                demo_df[col] = (demo_df[col] - mean_age) / std_age
                numerical_features.append(col)
                print(f"Age normalized: mean={mean_age:.2f}, std={std_age:.2f}")
                
            else:
                # Generic categorical handling
                if demo_df[col].dtype == 'object':
                    mode_value = demo_df[col].mode()[0] if not demo_df[col].mode().empty else 'UNKNOWN'
                    demo_df[col] = demo_df[col].fillna(mode_value)
                    categorical_features.append(col)
                else:
                    # Generic numerical handling
                    median_value = demo_df[col].median()
                    demo_df[col] = demo_df[col].fillna(median_value)
                    numerical_features.append(col)

        # Perform one-hot encoding for categorical variables
        
        encoded_features = []
        self.feature_names = []
        
        # Process categorical features with one-hot encoding
        for col in categorical_features:
            encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
            encoded = encoder.fit_transform(demo_df[[col]])
            
            # Get feature names
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]  # Skip first category
            self.feature_names.extend(feature_names)
            encoded_features.append(encoded)
            
            # Store encoder for potential future use
            if not hasattr(self, 'encoders'):
                self.encoders = {}
            self.encoders[col] = encoder
            
            print(f"{col} one-hot encoded: {len(feature_names)} features")
            print(f"  Categories: {encoder.categories_[0]}")
            print(f"  Features: {feature_names}")

        # Process numerical features
        for col in numerical_features:
            encoded_features.append(demo_df[[col]].values)
            self.feature_names.append(col)

        # Concatenate all features
        if encoded_features:
            all_features = np.concatenate(encoded_features, axis=1)
        else:
            all_features = np.array([]).reshape(len(demo_df), 0)

        # Store processed demographic data for each stay_id
        self.demographic_data = {}
        for i, stay_id in enumerate(demo_df.index):
            if stay_id in self.stay_ids:  # Only store data for stays in current split
                self.demographic_data[stay_id] = all_features[i].astype(np.float32)
        
        self.demo_feature_dim = all_features.shape[1]
        print(f"Total demographic feature dimension: {self.demo_feature_dim}")
        print(f"Feature names: {self.feature_names}")

    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]
        data = torch.FloatTensor(self.normalized_data[stay_id][:self.time_limit]) # [time_step,features]
        masks = torch.FloatTensor(self.missing_masks[stay_id][:self.time_limit])
        labels = torch.FloatTensor(np.atleast_1d(self.targets[idx])) 

        # 调试：检查数据形状
        if data.shape[0] != self.time_limit:
            print(f"Data shape mismatch for stay_id {stay_id}: expected {self.time_limit}, got {data.shape[0]}")
            print(f"Data shape: {data.shape}")
            print(f"Masks shape: {masks.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Stay ID: {stay_id}")
            print(f"Time limit: {self.time_limit}")
            print(f"Data: {self.normalized_data[stay_id][:self.time_limit]}")
            print(f"Masks: {self.missing_masks[stay_id][:self.time_limit]}")
            print(f"Labels: {self.targets[idx]}")
            raise ValueError(f"Data shape mismatch for stay_id {stay_id}")

        # Add demographic features if enabled
        if self.use_demographics:
            demo_features = torch.FloatTensor(self.demographic_data[stay_id])
            # 检查是否应该将demographics加入模型输入
            # 如果有 demographics_in_model_input 属性且为True，才加入模型输入
            if getattr(self, 'demographics_in_model_input', False):
                # Repeat demographic features for each time step
                demo_expanded = demo_features.unsqueeze(0).repeat(data.shape[0], 1)
                # Concatenate with EHR time series data
                data = torch.cat([data, demo_expanded], dim=1)
                # Extend masks for demographic features (always available, so mask = 1)
                demo_masks = torch.ones(masks.shape[0], self.demo_feature_dim)
                masks = torch.cat([masks, demo_masks], dim=1)

        # get image with potential dropout
        cxr_img = self._get_last_cxr_image_by_stay_id(stay_id, self.resized_base_path)
        has_cxr = False if cxr_img == None else True
        
        # Apply CXR dropout if enabled and this sample is selected for dropout
        if idx in self.dropped_cxr_samples:
            cxr_img = None
            has_cxr = False
            
        meta_attrs = self.meta_attr.loc[stay_id]
        cxr_time = self._get_last_cxr_time_by_stay_id(stay_id)

        # 添加CXR可用性到meta_attrs
        meta_attrs['has_cxr'] = has_cxr
        
        # Handle triplet format if needed
        if self.use_triplet:
            # Updated triplet logic - only using chartlab features since treatment_feature is not available
            chat_len = len(self.chartlab_feature)  # 27 features
            mask_len = len(self.mask_name)  # 22 masks (excluding rhythm_ features)
            demo_len = self.demo_feature_dim if self.use_demographics else 0
            timesteps = data.shape[0]
            
            # Split data into components
            chat_ts = data[:, :chat_len]  # [timesteps, 27] - chartlab features
            masks_ts = data[:, chat_len:chat_len+mask_len]  # [timesteps, 22] - masks for non-rhythm features
            if self.use_demographics:
                demo_ts = data[:, -demo_len:]
            
            # Prepare timestep indices
            time_indices = torch.arange(timesteps)
            
            # Generate chartlab triplets - only for features that have masks
            # Create mask for non-rhythm features only
            chat_mask = masks_ts != 0  # [timesteps, 22]
            
            # Get indices for non-rhythm features only
            non_rhythm_feature_indices = []
            for i, feat in enumerate(self.chartlab_feature):
                if not feat.startswith('rhythm_'):
                    non_rhythm_feature_indices.append(i)
            
            chat_feat_indices = torch.tensor(non_rhythm_feature_indices).expand(timesteps, mask_len)
            chat_triplets = torch.stack([
                time_indices.unsqueeze(1).expand_as(chat_mask)[chat_mask],
                chat_feat_indices[chat_mask],
                chat_ts[:, non_rhythm_feature_indices][chat_mask]
            ], dim=1)
            
            triplets = [chat_triplets]
            
            # Add demographic triplets if enabled
            # if self.use_demographics:
            #     demo_feat_indices = torch.arange(demo_len).expand(timesteps, demo_len) + chat_len
            #     demo_mask = torch.ones_like(demo_ts, dtype=bool)  # Demographics always available
            #     demo_triplets = torch.stack([
            #         time_indices.unsqueeze(1).expand_as(demo_mask)[demo_mask],
            #         demo_feat_indices[demo_mask],
            #         demo_ts[demo_mask]
            #     ], dim=1)
            #     triplets.append(demo_triplets)
            
            # Combine all triplets and sort by timestep
            data = torch.cat(triplets, dim=0)
            data = data[data[:, 0].argsort()]

        # Handle different tasks for one-hot encoding and label format
        if self.one_hot:
            if self.task == "mortality":
                num_classes = 2 
                labels_one_hot = torch.zeros(num_classes)  
                labels_one_hot.scatter_(0, labels.long(), 1)  
                return stay_id, data, masks, cxr_img, has_cxr, labels_one_hot, meta_attrs, torch.LongTensor([idx]), cxr_time
            elif self.task == "los":
                num_classes = 7  # 7 LoS bins (0-6)，去掉了0-2天的类别
                labels_one_hot = torch.zeros(num_classes)  
                labels_one_hot.scatter_(0, labels.long(), 1)  
                return stay_id, data, masks, cxr_img, has_cxr, labels_one_hot, meta_attrs, torch.LongTensor([idx]), cxr_time
        
        # For LoS task, ensure labels are in correct format (integer for classification)
        if self.task == "los":
            labels = labels.long()  # Convert to long tensor for classification
        
        assert not np.isnan(data).any(), f"NaN in data for stay_id {stay_id}"
        assert not np.isnan(labels).any(), f"NaN in labels for stay_id {stay_id}"
        
        return stay_id, data, masks, cxr_img, has_cxr, labels, meta_attrs, torch.LongTensor([idx]), cxr_time

    def __len__(self):
        return len(self.stay_ids)

    def __load_time_series_by_stay_id(self, stay_id):
        stay_data_origin = pd.read_csv(self.data_root/'merged'/f'{stay_id}.csv').sort_values(by='timestep')
        stay_data = stay_data_origin[['timestep'] + self.features]  # Extract needed features
        stay_data_mask = stay_data_origin[self.mask_name]  # mask data

        # 创建完整的时间步序列 (0 到 time_limit-1)
        complete_timesteps = pd.DataFrame({'timestep': range(self.time_limit)})
        
        # 将原始数据与完整时间步合并，缺失的时间步会自动填充 NaN
        stay_data = complete_timesteps.merge(stay_data, on='timestep', how='left')
        stay_data_mask = complete_timesteps.merge(
            stay_data_origin[['timestep'] + self.mask_name], 
            on='timestep', 
            how='left'
        )

        # 识别完全缺失的时间步
        missing_timesteps = ~complete_timesteps['timestep'].isin(stay_data_origin['timestep'])
        
        # 对于完全缺失的时间步，直接将mask设为1（表示缺失）
        if self.mask_name:
            for mask_col in self.mask_name:
                stay_data_mask.loc[missing_timesteps, mask_col] = 1.0
            
            # 对于存在的时间步中的NaN mask值，填充为1
            stay_data_mask[self.mask_name] = stay_data_mask[self.mask_name].fillna(1.0)

        # 分离 rhythm_ 特征和其他特征
        rhythm_features = [feat for feat in self.features if feat.startswith('rhythm_')]
        non_rhythm_features = [feat for feat in self.features if not feat.startswith('rhythm_')]
        
        # 计算 missing mask BEFORE imputation（重要：在填充之前计算）
        if non_rhythm_features:
            # 对于完全缺失的时间步，所有特征的mask都是1（表示该时间步是插补的）
            original_mask = stay_data[non_rhythm_features].isna().astype(float)
            original_mask.loc[missing_timesteps, :] = 1.0
            missing_mask = original_mask.values
        else:
            # 如果没有非rhythm特征，创建空的mask
            missing_mask = np.zeros((len(stay_data), 0))
        
        # 对于 rhythm_ 特征：直接填充 0，不使用 ffill
        if rhythm_features:
            stay_data[rhythm_features] = stay_data[rhythm_features].fillna(0.0)
        
        # 对于非 rhythm_ 特征：使用 ffill + 默认值填充
        if non_rhythm_features:
            stay_data[non_rhythm_features] = stay_data[non_rhythm_features].ffill().fillna(
                {feat: self.default_imputation[feat] for feat in non_rhythm_features}
            )
        
        # 合并所有特征
        data_imputed = stay_data[self.features]
        
        # 验证没有NaN值
        assert not data_imputed.isna().any().any(), f"NaN still exists in data_imputed for stay_id {stay_id}"
        
        # robust normalization - 只对需要标准化的特征进行处理
        data_normalized = data_imputed.copy()

        for i, feat in enumerate(self.features):
            if feat in self.features_no_normalization:
                # 不需要标准化的特征直接保留原值
                continue
            else:
                # 需要标准化的特征进行 robust normalization
                median_val = self.features_stats['median'][i]
                iqr_val = self.features_stats['iqr'][i]
                
                # 安全检查
                if pd.isna(median_val) or pd.isna(iqr_val):
                    print(f"Warning: NaN stats for feature {feat}: median={median_val}, iqr={iqr_val}")
                    # 如果统计值为NaN，保留原值
                    continue
                elif iqr_val == 0:
                    # IQR为0时，使用标准差进行标准化，如果标准差也为0则进行中心化
                    std_val = self.features_stats['std'][i]
                    if pd.isna(std_val) or std_val == 0:
                        # 标准差也为0，只进行中心化（减去均值）
                        data_normalized.iloc[:, i] = data_imputed.iloc[:, i] - median_val
                        # print(f"Info: Feature {feat} has IQR=0 and std=0, applying centering only")
                    else:
                        # 使用标准差进行标准化
                        data_normalized.iloc[:, i] = (data_imputed.iloc[:, i] - median_val) / std_val
                        # print(f"Info: Feature {feat} has IQR=0, using std normalization instead")
                else:
                    # 正常的robust normalization
                    data_normalized.iloc[:, i] = (data_imputed.iloc[:, i] - median_val) / iqr_val

        # 验证标准化后没有NaN值
        assert not data_normalized.isna().any().any(), f"NaN in data_normalized for stay_id {stay_id}"
        
        concatenated_data = np.concatenate((data_normalized.values, stay_data_mask[self.mask_name].values), axis=1)
        
        # 最终检查
        assert not np.isnan(concatenated_data).any(), f"NaN in concatenated_data for stay_id {stay_id}"

        return stay_id, concatenated_data, missing_mask
    

    def load_and_normalize_time_series(self):
        normalized_data = {}
        missing_masks = {}

        for stay_id in tqdm(self.stay_ids, desc='Loading and pre-processing raw time series'):
            _, data, masks = self.__load_time_series_by_stay_id(stay_id)
            normalized_data[stay_id] = data
            missing_masks[stay_id] = masks

        return normalized_data, missing_masks

    def _get_last_cxr_time_by_stay_id(self, stay_id):
        valid_cxrs = self.ehr_meta.loc[self.ehr_meta['stay_id'] == stay_id, 'valid_cxrs'].values[0]
        # "[('6b47030f-7100fb41-5d481485-947a91e3-71c54cbe', Timestamp('2142-07-05 20:12:11'))]"
        # no cxr data
        if pd.isna(valid_cxrs) or valid_cxrs == "[]":
            return 0.0

        valid_cxrs_clean = re.sub(r"Timestamp\('([^']+)'\)", r"'\1'", valid_cxrs)
        valid_cxrs_clean_parse = ast.literal_eval(valid_cxrs_clean)
        cxr_time = valid_cxrs_clean_parse[-1][1]

        intime=self.ehr_meta.loc[self.ehr_meta['stay_id'] == stay_id, 'intime'].values[0]
        # 2158-09-07 21:44:00
        cxr_time=self._get_hours_diff(intime,cxr_time)
        return cxr_time
        
    

    def _get_hours_diff(self, timestamp1, timestamp2):
       
        dt1 = datetime.strptime(timestamp1, '%Y-%m-%d %H:%M:%S')
        dt2 = datetime.strptime(timestamp2, '%Y-%m-%d %H:%M:%S')
        
        
        delta = dt2 - dt1
        return delta.total_seconds() / 3600    

    def _get_last_cxr_image_by_stay_id(self, stay_id, resized_base_path='/research/mimic_cxr_resized'):
        valid_cxrs = self.ehr_meta.loc[self.ehr_meta['stay_id'] == stay_id, 'valid_cxrs'].values[0]
        # "[('6b47030f-7100fb41-5d481485-947a91e3-71c54cbe', Timestamp('2142-07-05 20:12:11'))]"
        # no cxr data
        if pd.isna(valid_cxrs) or valid_cxrs == "[]":
            return None

        # re parse dicom_id
        valid_cxrs_clean = re.sub(r"Timestamp\('([^']+)'\)", r"'\1'", valid_cxrs)
        valid_cxrs_clean_parse = ast.literal_eval(valid_cxrs_clean)
        dicom_id = valid_cxrs_clean_parse[-1][0]

        subject_id = self.ehr_meta.loc[self.ehr_meta['stay_id'] == stay_id, 'subject_id'].values[0]
        img_path = self.get_image_path(dicom_id, subject_id, resized_base_path=resized_base_path)
        if img_path is None:
            return None

        try:
            cxr_img = Image.open(img_path).convert('RGB')
            return self.cxr_transform(cxr_img)
        except FileNotFoundError:
            print(f"{img_path} not exists!!!!")
            return None

    def get_image_path(self, dicom_id, subject_id, resized_base_path='/research/mimic_cxr_resized'):
        image_path = f"{resized_base_path}/{dicom_id}.jpg"
        return image_path


def pad_temporal_data(batch):
    """Collate function for clinical tasks"""
    stay_ids, data, masks, cxr_imgs, has_cxr, labels, meta_attrs, idx, cxr_times = zip(*batch)
    seq_len = [x.shape[0] for x in data]  # Get time steps
    max_len = max(seq_len)
    
    # 检查是否所有序列长度都一致（应该都是time_limit长度）
    if len(set(seq_len)) == 1:
        # 所有样本长度一致，直接stack
        data_padded = torch.stack(data, dim=0)
        masks_padded = torch.stack(masks, dim=0)
        # print(f"All sequences have same length: {seq_len[0]}")
    else:
        # 仍有不同长度的序列，进行padding
        # print(f"Sequence lengths vary: {set(seq_len)}, padding to {max_len}")
        data_padded = torch.stack([torch.cat([x, torch.zeros(max_len-x.shape[0], x.shape[1])], dim=0)
                                   for x in data], dim=0)
        masks_padded = torch.stack([torch.cat([x, torch.zeros(max_len-x.shape[0], x.shape[1])], dim=0)
                                   for x in masks], dim=0)

    # Handle CXR images
    processed_cxr_imgs = []
    for x in cxr_imgs:
        if x is None:
            processed_cxr_imgs.append(torch.zeros(3, 224, 224))
        else:
            if isinstance(x, tuple):
                processed_cxr_imgs.append(torch.tensor(x))
            else:
                processed_cxr_imgs.append(x)

    cxr_imgs = torch.stack(processed_cxr_imgs)
    has_cxr = torch.FloatTensor(has_cxr)
    labels = torch.stack(labels, dim=0)
    idx = torch.stack(idx, dim=0)
    cxr_times=torch.FloatTensor(cxr_times)
    meta_attrs = pd.DataFrame(meta_attrs)
    
    batch_data = {
        'stay_ids': list(stay_ids),
        'seq_len': seq_len,
        'ehr_ts': data_padded,
        'ehr_masks': masks_padded,
        'cxr_imgs': cxr_imgs,
        'has_cxr': has_cxr,
        'labels': labels,
        'meta_attrs': meta_attrs,
        'idx': idx,
        'cxr_times':cxr_times,
    }
    return batch_data


def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loaders(ehr_data_dir, task, replication, batch_size,
                        num_workers, time_limit=None, matched_subset=False, use_triplet=False, seed=None, one_hot=False, pkl_dir=None,
                        resized_base_path='/research/mimic_cxr_resized',
                        image_meta_path="/hdd/datasets/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv",
                        train_matched=None, val_matched=None, test_matched=None,
                        use_demographics=False, demographic_cols=None,
                        use_label_weights=False, label_weight_method='balanced', custom_label_weights=None,
                        cxr_dropout_rate=0.0, cxr_dropout_seed=None, demographics_in_model_input=False):
    """Create data loaders for clinical tasks with flexible matched/full data selection and optional demographics"""
    set_seed(seed)
    time_limit = 48  # Fixed time limit to 48 hours
    print(f"Time limit is {time_limit} hours")
    print(f"Task: {task}")
    
    # Handle different parameter sources
    if train_matched is None:
        train_matched = matched_subset
    if val_matched is None:
        val_matched = matched_subset  
    if test_matched is None:
        test_matched = matched_subset
    
    print(f"Data subset configuration:")
    print(f"  Train: {'matched' if train_matched else 'full'} data")
    print(f"  Validation: {'matched' if val_matched else 'full'} data") 
    print(f"  Test: {'matched' if test_matched else 'full'} data")
    
    if use_demographics:
        demo_cols = demographic_cols if demographic_cols else ['age', 'gender', 'admission_type', 'race']
        print(f"  Demographics: enabled with columns {demo_cols}")
    else:
        print(f"  Demographics: disabled")
    
    if use_label_weights:
        print(f"  Label weights: enabled with method '{label_weight_method}'")
        if custom_label_weights:
            print(f"  Custom weights: {custom_label_weights}")
    else:
        print(f"  Label weights: disabled")
    
    if cxr_dropout_rate > 0.0:
        print(f"  CXR dropout: enabled with rate {cxr_dropout_rate} (seed: {cxr_dropout_seed})")
    else:
        print(f"  CXR dropout: disabled")
    
    # Validate task type
    if task not in ['mortality', 'phenotype', 'los']:
        raise ValueError(f"Unknown task type: {task}. Supported tasks: mortality, phenotype, los")
    
    # Create data loaders for clinical tasks
    return _create_clinical_loaders(
        ehr_data_dir=ehr_data_dir,
        task=task,
        replication=replication,
        batch_size=batch_size,
        num_workers=num_workers,
        time_limit=time_limit,
        train_matched=train_matched,
        val_matched=val_matched,
        test_matched=test_matched,
        use_triplet=use_triplet, 
        seed=seed,
        one_hot=one_hot,
        resized_base_path=resized_base_path,
        image_meta_path=image_meta_path,
        pkl_dir=pkl_dir,
        use_demographics=use_demographics,
        demographic_cols=demographic_cols,
        use_label_weights=use_label_weights,
        label_weight_method=label_weight_method,
        custom_label_weights=custom_label_weights,
        cxr_dropout_rate=cxr_dropout_rate,
        cxr_dropout_seed=cxr_dropout_seed,
        demographics_in_model_input=demographics_in_model_input
    )

def _create_clinical_loaders(**kwargs):
    """Create data loaders for clinical tasks with demographics and label weights support"""
    data_loaders = []
    split_configs = [
        ('train', kwargs['train_matched']),
        ('val', kwargs['val_matched']),
        ('test', kwargs['test_matched'])
    ]
    
    for split, matched_subset in split_configs:
        is_train = (split == 'train')
        
        print(f"Creating {split} dataset with {'matched' if matched_subset else 'full'} data")
        
        ds = MultiModalMIMIC(
            seed=kwargs['seed'],
            data_root=kwargs['ehr_data_dir'],
            fold=kwargs['replication'], 
            partition=split,
            task=kwargs['task'],
            time_limit=kwargs['time_limit'],
            matched_subset=matched_subset,
            use_triplet=kwargs['use_triplet'],
            one_hot=kwargs['one_hot'],
            resized_base_path=kwargs['resized_base_path'],
            image_meta_path=kwargs['image_meta_path'],
            pkl_dir=kwargs['pkl_dir'],
            use_demographics=kwargs['use_demographics'],
            demographic_cols=kwargs['demographic_cols'],
            use_label_weights=kwargs.get('use_label_weights', False),
            label_weight_method=kwargs.get('label_weight_method', 'balanced'),
            custom_label_weights=kwargs.get('custom_label_weights', None),
            cxr_dropout_rate=kwargs.get('cxr_dropout_rate', 0.0),
            cxr_dropout_seed=kwargs.get('cxr_dropout_seed', None),
            demographics_in_model_input=kwargs.get('demographics_in_model_input', False)
        )
        
        g = torch.Generator()
        g.manual_seed(kwargs['seed'])
        
        dl = DataLoader(
            ds,
            batch_size=kwargs['batch_size'],
            pin_memory=True,
            shuffle=is_train,
            drop_last=is_train,
            num_workers=kwargs['num_workers'],
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=pad_temporal_data
        )
        
        data_loaders.append(dl)
    
    return data_loaders
