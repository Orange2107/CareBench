"""
Fairness Metrics for Clinical Models

This module implements fairness evaluation metrics including:
- Group fairness metrics (PRAUC gap, worse-case PRAUC)
- Max-min fairness analysis
- Demographic parity metrics

The metrics are computed across demographic groups (race, gender, age groups, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
import warnings


class FairnessEvaluator:
    """
    Evaluator for fairness metrics across demographic groups
    """
    
    def __init__(self, 
                 sensitive_attributes: List[str] = None,
                 age_bins: List[float] = None,
                 compute_intersectional: bool = True,
                 include_cxr_availability: bool = True):  # 默认启用
        """
        Initialize fairness evaluator
        
        Args:
            sensitive_attributes: List of sensitive attribute columns to evaluate
            age_bins: Bins for age discretization if age is included
            compute_intersectional: Whether to compute intersectional fairness metrics
            include_cxr_availability: Whether to include CXR availability as a sensitive attribute
        """
        if sensitive_attributes is None:
            sensitive_attributes = ['race', 'gender', 'age']
        
        # 如果启用CXR可用性分析，添加到sensitive attributes
        if include_cxr_availability and 'has_cxr' not in sensitive_attributes:
            sensitive_attributes = sensitive_attributes + ['has_cxr']
        
        self.sensitive_attributes = sensitive_attributes
        self.compute_intersectional = compute_intersectional
        self.include_cxr_availability = include_cxr_availability
        
        # Default age bins: [0, 20, 40, 60, 80, 80+]
        if age_bins is None:
            age_bins = [0, 20, 40, 60, 80, float('inf')]
        self.age_bins = age_bins
        
        # Age bin labels
        self.age_labels = []
        for i in range(len(age_bins) - 1):
            lower = age_bins[i]
            upper = age_bins[i + 1]
            
            # Check if bins are infinity
            is_lower_inf = (lower == float('inf') or lower == float('-inf'))
            is_upper_inf = (upper == float('inf') or upper == float('-inf'))
            
            # Skip if lower bound is infinity (shouldn't happen)
            if is_lower_inf:
                continue
            
            # Create label based on upper bound
            if is_upper_inf:
                self.age_labels.append(f"{int(lower)}+")
            else:
                self.age_labels.append(f"{int(lower)}-{int(upper)}")
        
        # Define ethnicity mapping to 5 main categories (same as dataset.py)
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

    def _discretize_age(self, ages: np.ndarray) -> np.ndarray:
        """
        Discretize continuous age values into bins
        
        Args:
            ages: Array of age values
            
        Returns:
            Array of age bin labels
        """
        age_discrete = np.digitize(ages, self.age_bins[1:-1])
        age_labels = [self.age_labels[i] for i in age_discrete]
        return np.array(age_labels)

    def _map_ethnicity(self, ethnicity_values: np.ndarray) -> np.ndarray:
        """
        Map detailed ethnicity values to 5 main categories
        
        Args:
            ethnicity_values: Array of detailed ethnicity values
            
        Returns:
            Array of mapped ethnicity categories
        """
        mapped_values = []
        for value in ethnicity_values:
            if pd.isna(value) or value == '' or value is None:
                mapped_values.append('OTHER')
            else:
                mapped_values.append(self.ethnicity_mapping.get(str(value).upper(), 'OTHER'))
        return np.array(mapped_values)

    def _prepare_demographic_data(self, meta_attrs: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare demographic data for fairness evaluation
        """
        demo_df = meta_attrs.copy()
        
        # Process age if it's in sensitive attributes
        if 'age' in self.sensitive_attributes and 'age' in demo_df.columns:
            demo_df['age_group'] = self._discretize_age(demo_df['age'].values)
            # Replace 'age' with 'age_group' in sensitive attributes
            sensitive_attrs = [attr if attr != 'age' else 'age_group' 
                             for attr in self.sensitive_attributes]
            self.sensitive_attributes = sensitive_attrs
        
        # Process CXR availability if it's in sensitive attributes
        if 'has_cxr' in self.sensitive_attributes and 'has_cxr' in demo_df.columns:
            print("Processing CXR availability for fairness analysis...")
            
            # Convert boolean to string for consistency
            demo_df['has_cxr'] = demo_df['has_cxr'].astype(str)
            
            # Show distribution
            cxr_counts = demo_df['has_cxr'].value_counts()
            print(f"CXR availability distribution:")
            for availability, count in cxr_counts.items():
                print(f"  {availability}: {count} ({count/len(demo_df)*100:.1f}%)")
        
        # Process race/ethnicity mapping if it's in sensitive attributes
        race_columns = ['race', 'ethnicity', 'race_ethnicity']
        for race_col in race_columns:
            if race_col in self.sensitive_attributes and race_col in demo_df.columns:
                print(f"Mapping {race_col} values to 5 main categories...")
                original_values = demo_df[race_col].unique()
                print(f"Original {race_col} values: {sorted([str(v) for v in original_values if pd.notna(v)])}")
                
                demo_df[race_col] = self._map_ethnicity(demo_df[race_col].values)
                
                mapped_values = demo_df[race_col].unique()
                print(f"Mapped {race_col} values: {sorted(mapped_values)}")
                
                # Show mapping counts
                value_counts = demo_df[race_col].value_counts()
                print(f"{race_col} distribution after mapping:")
                for category, count in value_counts.items():
                    print(f"  {category}: {count} ({count/len(demo_df)*100:.1f}%)")
        
        # Handle missing values by creating an "Unknown" category
        for attr in self.sensitive_attributes:
            if attr in demo_df.columns:
                demo_df[attr] = demo_df[attr].fillna('Unknown')
        
        return demo_df

    def compute_group_metrics(self, 
                            y_true: np.ndarray,
                            y_score: np.ndarray, 
                            meta_attrs: pd.DataFrame,
                            task_type: str = 'binary') -> Dict[str, Any]:
        """
        Compute fairness metrics across demographic groups
        
        Args:
            y_true: True labels
            y_score: Predicted scores/probabilities
            meta_attrs: DataFrame with demographic attributes
            task_type: Type of task ('binary', 'multiclass', 'multilabel')
            
        Returns:
            Dictionary containing fairness metrics
        """
        if len(y_true) != len(meta_attrs):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs meta_attrs ({len(meta_attrs)})")
        
        # Prepare demographic data
        demo_df = self._prepare_demographic_data(meta_attrs)
        
        fairness_results = {}
        
        # Compute metrics for each sensitive attribute
        has_cxr_values = demo_df['has_cxr'].values if 'has_cxr' in demo_df.columns else None
        for attr in self.sensitive_attributes:
            if attr not in demo_df.columns:
                print(f"Warning: Sensitive attribute '{attr}' not found in demographic data")
                continue
                
            attr_results = self._compute_attribute_fairness(
                y_true, 
                y_score, 
                demo_df[attr].values, 
                attr, 
                task_type,
                has_cxr_values=has_cxr_values
            )
            fairness_results.update(attr_results)
        
        # Compute intersectional fairness if requested
        if self.compute_intersectional and len(self.sensitive_attributes) >= 2:
            intersectional_results = self._compute_intersectional_fairness(
                y_true, y_score, demo_df, task_type
            )
            fairness_results.update(intersectional_results)
        
        return fairness_results

    def _compute_attribute_fairness(self, 
                                  y_true: np.ndarray,
                                  y_score: np.ndarray,
                                  attribute_values: np.ndarray,
                                  attribute_name: str,
                                  task_type: str,
                                  has_cxr_values: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute fairness metrics for a single sensitive attribute
        
        Args:
            y_true: True labels
            y_score: Predicted scores
            attribute_values: Values of the sensitive attribute
            attribute_name: Name of the sensitive attribute
            task_type: Type of task
            
        Returns:
            Dictionary with fairness metrics for this attribute
        """
        results = {}
        
        # Get unique groups
        unique_groups = np.unique(attribute_values)
        
        if len(unique_groups) < 2:
            print(f"Warning: Only {len(unique_groups)} group(s) found for attribute '{attribute_name}'")
            return results
        
        group_metrics = {}
        group_sizes = {}
        group_missing_rates = {}
        
        # Compute metrics for each group
        for group in unique_groups:
            mask = attribute_values == group
            if np.sum(mask) < 10:  # Skip groups with too few samples
                print(f"Warning: Group '{group}' in '{attribute_name}' has only {np.sum(mask)} samples, skipping")
                continue
                
            y_true_group = y_true[mask]
            y_score_group = y_score[mask]
            
            try:
                if task_type == 'binary':
                    metrics = self._compute_binary_metrics(y_true_group, y_score_group)
                elif task_type == 'multiclass':
                    metrics = self._compute_multiclass_metrics(y_true_group, y_score_group)
                elif task_type == 'multilabel':
                    metrics = self._compute_multilabel_metrics(y_true_group, y_score_group)
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
                
                group_metrics[group] = metrics
                group_sizes[group] = np.sum(mask)
                
                # Missing modality rate for this group if has_cxr is available
                if has_cxr_values is not None:
                    group_missing_rates[group] = float(np.mean(has_cxr_values[mask] == 0))
                
                # Store individual group metrics
                for metric_name, value in metrics.items():
                    results[f'fairness/{attribute_name}/{group}/{metric_name}'] = float(value)
                    
            except Exception as e:
                print(f"Warning: Error computing metrics for group '{group}' in '{attribute_name}': {e}")
                continue
        
        if len(group_metrics) < 2:
            print(f"Warning: Not enough valid groups for fairness analysis of '{attribute_name}'")
            return results
        
        # Compute fairness metrics
        fairness_metrics = self._compute_fairness_statistics(group_metrics, attribute_name)
        results.update(fairness_metrics)

        # Compute correlation between subgroup fairness gaps and missing-modality rate
        if has_cxr_values is not None and len(group_missing_rates) >= 2:
            for metric_name in next(iter(group_metrics.values())).keys():
                metric_values = []
                missing_rates = []
                group_names = []
                for group, metrics in group_metrics.items():
                    if metric_name in metrics and group in group_missing_rates:
                        metric_values.append(metrics[metric_name])
                        missing_rates.append(group_missing_rates[group])
                        group_names.append(group)

                if len(metric_values) < 2:
                    continue

                metric_values = np.array(metric_values, dtype=float)
                missing_rates = np.array(missing_rates, dtype=float)

                # Fairness gap for each group: deviation from group mean
                gaps = metric_values - metric_values.mean()

                # Pearson correlation between gaps and missing-modality rates
                if np.std(gaps) > 0 and np.std(missing_rates) > 0:
                    corr = float(np.corrcoef(gaps, missing_rates)[0, 1])
                else:
                    corr = 0.0

                results[f'fairness/{attribute_name}/{metric_name}/pearson_correlation_gap_missing_modality'] = corr
        
        # Add group size information
        total_samples = sum(group_sizes.values())
        for group, size in group_sizes.items():
            results[f'fairness/{attribute_name}/{group}/sample_size'] = int(size)
            results[f'fairness/{attribute_name}/{group}/sample_proportion'] = float(size / total_samples)
        
        return results

    def _compute_binary_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        """Compute metrics for binary classification - PRAUC, equal odds, and calibration"""
        y_true = y_true.flatten()
        y_score = y_score.flatten()
        
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            # Only one class present, return limited metrics
            return {
                'sample_positive_rate': float(np.mean(y_true)),
            }
        
        # PRAUC
        try:
            prauc = average_precision_score(y_true, y_score)
        except ValueError:
            prauc = 0.0

        # Equal odds (TPR/FPR gap will be computed across groups later)
        threshold = 0.5
        y_pred = (y_score >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall / sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Calibration error (ECE-style, 10 bins)
        num_bins = 10
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        for i in range(num_bins):
            mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1])
            if not np.any(mask):
                continue
            avg_conf = np.mean(y_score[mask])
            avg_label = np.mean(y_true[mask])
            bin_frac = np.mean(mask)
            ece += bin_frac * abs(avg_conf - avg_label)
        
        return {
            'PRAUC': float(prauc),
            'TPR': float(tpr),
            'FPR': float(fpr),
            'ECE': float(ece),
            'sample_positive_rate': float(np.mean(y_true)),
        }

    def _compute_multiclass_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        """Compute metrics for multiclass classification - accuracy, TPR, FPR, and ECE"""
        y_true = y_true.flatten().astype(int)
        
        if y_score.ndim > 1 and y_score.shape[1] > 1:
            y_pred = np.argmax(y_score, axis=1)
            # Get predicted probabilities for the predicted class
            y_score_max = np.max(y_score, axis=1)
        else:
            y_pred = y_score.flatten().astype(int)
            y_score_max = y_score.flatten()
        
        # Accuracy
        acc = float(accuracy_score(y_true, y_pred))
        
        # For multiclass, compute TPR and FPR using one-vs-rest approach
        # TPR (recall) and FPR are computed per class, then averaged
        num_classes = len(np.unique(y_true))
        if num_classes < 2:
            return {
                'accuracy': acc,
            }
        
        tpr_scores = []
        fpr_scores = []
        
        for class_idx in range(num_classes):
            # Create binary labels for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            # Compute TP, FP, FN, TN for this class
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            tpr_scores.append(tpr)
            fpr_scores.append(fpr)
        
        avg_tpr = float(np.mean(tpr_scores)) if tpr_scores else 0.0
        avg_fpr = float(np.mean(fpr_scores)) if fpr_scores else 0.0
        
        # Calibration error (ECE-style, 10 bins) for multiclass
        # Use max probability as confidence score
        num_bins = 10
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        
        # ECE: bin by max probability (confidence)
        for i in range(num_bins):
            mask = (y_score_max >= bin_edges[i]) & (y_score_max < bin_edges[i + 1])
            if not np.any(mask):
                continue
            # Average confidence in this bin
            avg_conf = np.mean(y_score_max[mask])
            # Average accuracy in this bin (calibration)
            avg_acc = np.mean((y_true[mask] == y_pred[mask]).astype(float))
            bin_frac = np.mean(mask)
            ece += bin_frac * abs(avg_conf - avg_acc)
        
        return {
            'accuracy': acc,
            'TPR': avg_tpr,
            'FPR': avg_fpr,
            'ECE': float(ece),
        }

    def _compute_multilabel_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        """Compute metrics for multilabel classification - PRAUC, equal odds, and calibration"""
        
        num_labels = y_true.shape[1]
        prauc_scores = []
        tpr_scores = []
        fpr_scores = []
        ece_scores = []

        threshold = 0.5
        num_bins = 10
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        
        for i in range(num_labels):
            y_t = y_true[:, i]
            y_s = y_score[:, i]

            # Skip labels with only one class present
            if len(np.unique(y_t)) < 2:
                continue

            # PRAUC
            try:
                prauc = average_precision_score(y_t, y_s)
                prauc_scores.append(prauc)
            except ValueError:
                pass

            # Equal odds (per-label TPR/FPR, then average over labels)
            y_pred = (y_s >= threshold).astype(int)
            tp = np.sum((y_t == 1) & (y_pred == 1))
            fp = np.sum((y_t == 0) & (y_pred == 1))
            fn = np.sum((y_t == 1) & (y_pred == 0))
            tn = np.sum((y_t == 0) & (y_pred == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr_scores.append(tpr)
            fpr_scores.append(fpr)

            # Calibration (ECE-style) per label
            ece = 0.0
            for b in range(num_bins):
                mask = (y_s >= bin_edges[b]) & (y_s < bin_edges[b + 1])
                if not np.any(mask):
                    continue
                avg_conf = np.mean(y_s[mask])
                avg_label = np.mean(y_t[mask])
                bin_frac = np.mean(mask)
                ece += bin_frac * abs(avg_conf - avg_label)
            ece_scores.append(ece)
        
        avg_prauc = np.mean(prauc_scores) if prauc_scores else 0.0
        avg_tpr = np.mean(tpr_scores) if tpr_scores else 0.0
        avg_fpr = np.mean(fpr_scores) if fpr_scores else 0.0
        avg_ece = np.mean(ece_scores) if ece_scores else 0.0
        
        return {
            'PRAUC': float(avg_prauc),
            'TPR': float(avg_tpr),
            'FPR': float(avg_fpr),
            'ECE': float(avg_ece),
        }

    def _compute_fairness_statistics(self, 
                                   group_metrics: Dict[str, Dict[str, float]], 
                                   attribute_name: str) -> Dict[str, float]:
        """
        Compute fairness statistics across groups
        
        Args:
            group_metrics: Dictionary mapping group names to their metrics
            attribute_name: Name of the sensitive attribute
            
        Returns:
            Dictionary with fairness statistics
        """
        results = {}
        
        # Get all metric names
        all_metrics = set()
        for metrics in group_metrics.values():
            all_metrics.update(metrics.keys())
        
        for metric_name in all_metrics:
            metric_values = []
            group_names = []
            
            for group, metrics in group_metrics.items():
                if metric_name in metrics:
                    metric_values.append(metrics[metric_name])
                    group_names.append(group)
            
            if len(metric_values) < 2:
                continue
            
            metric_values = np.array(metric_values)
            
            # Overall statistics
            results[f'fairness/{attribute_name}/{metric_name}/mean'] = float(np.mean(metric_values))
            results[f'fairness/{attribute_name}/{metric_name}/std'] = float(np.std(metric_values))
            results[f'fairness/{attribute_name}/{metric_name}/min'] = float(np.min(metric_values))
            results[f'fairness/{attribute_name}/{metric_name}/max'] = float(np.max(metric_values))
            
            # Fairness-specific metrics
            max_val = np.max(metric_values)
            min_val = np.min(metric_values)
            
            # Gap metrics
            results[f'fairness/{attribute_name}/{metric_name}/gap'] = float(max_val - min_val)
            results[f'fairness/{attribute_name}/{metric_name}/ratio'] = float(min_val / max_val) if max_val > 0 else 0.0
            
            # Worst-case and best-case groups
            worst_idx = np.argmin(metric_values)
            best_idx = np.argmax(metric_values)
            
            results[f'fairness/{attribute_name}/{metric_name}/worst_group'] = group_names[worst_idx]
            results[f'fairness/{attribute_name}/{metric_name}/best_group'] = group_names[best_idx]
            results[f'fairness/{attribute_name}/{metric_name}/worst_case'] = float(min_val)
            results[f'fairness/{attribute_name}/{metric_name}/best_case'] = float(max_val)
            
            # Max-min fairness (worst-case performance)
            results[f'fairness/{attribute_name}/{metric_name}/max_min_fairness'] = float(min_val)
        
        return results

    def _compute_intersectional_fairness(self, 
                                       y_true: np.ndarray,
                                       y_score: np.ndarray,
                                       demo_df: pd.DataFrame,
                                       task_type: str) -> Dict[str, float]:
        """
        Compute intersectional fairness metrics
        """
        results = {}
        
        # Create intersectional groups (combinations of attributes)
        available_attrs = [attr for attr in self.sensitive_attributes if attr in demo_df.columns]
        
        if len(available_attrs) < 2:
            return results
        
        # Specifically compute age x race intersectional fairness if both are available
        age_attr = None
        race_attr = None
        
        # Find age and race attributes
        for attr in available_attrs:
            if 'age' in attr.lower():
                age_attr = attr
            elif 'race' in attr.lower() or 'ethnicity' in attr.lower():
                race_attr = attr
        
        if age_attr and race_attr:
            print(f"Computing intersectional fairness for {age_attr} x {race_attr}")
            
            # Create combined attribute for age x race
            combined_attr = demo_df[age_attr].astype(str) + "_x_" + demo_df[race_attr].astype(str)
            
            # Count combinations to see which ones have enough samples
            combination_counts = combined_attr.value_counts()
            print(f"Age x Race intersectional groups:")
            for combo, count in combination_counts.items():
                print(f"  {combo}: {count} samples")
            
            intersectional_results = self._compute_attribute_fairness(
                y_true, y_score, combined_attr.values, f"{age_attr}_x_{race_attr}", task_type
            )
            
            results.update(intersectional_results)
        
        # Also compute all other pairs of attributes as before
        for i in range(len(available_attrs)):
            for j in range(i + 1, len(available_attrs)):
                attr1, attr2 = available_attrs[i], available_attrs[j]
                
                # Skip if we already computed age x race above
                if (age_attr and race_attr and 
                    ((attr1 == age_attr and attr2 == race_attr) or 
                     (attr1 == race_attr and attr2 == age_attr))):
                    continue
                
                # Create combined attribute
                combined_attr = demo_df[attr1].astype(str) + "_x_" + demo_df[attr2].astype(str)
                
                intersectional_results = self._compute_attribute_fairness(
                    y_true, y_score, combined_attr.values, f"{attr1}_x_{attr2}", task_type
                )
                
                results.update(intersectional_results)
        
        return results


def compute_fairness_metrics(y_true: np.ndarray,
                           y_score: np.ndarray,
                           meta_attrs: pd.DataFrame,
                           task_type: str = 'binary',
                           sensitive_attributes: List[str] = None,
                           age_bins: List[float] = None,
                           compute_intersectional: bool = False,
                           include_cxr_availability: bool = True) -> Tuple[Dict[str, Any], str]:  # 默认启用
    """
    Convenience function to compute fairness metrics
    """
    print(f"=== FAIRNESS METRICS DEBUG ===")
    print(f"y_true shape: {y_true.shape}")
    print(f"y_score shape: {y_score.shape}")
    print(f"meta_attrs shape: {meta_attrs.shape}")
    print(f"meta_attrs columns: {meta_attrs.columns.tolist()}")
    print(f"sensitive_attributes: {sensitive_attributes}")
    print(f"task_type: {task_type}")
    print(f"include_cxr_availability: {include_cxr_availability}")
    print(f"=== END DEBUG ===")
    
    evaluator = FairnessEvaluator(
        sensitive_attributes=sensitive_attributes,
        age_bins=age_bins,
        compute_intersectional=compute_intersectional,
        include_cxr_availability=include_cxr_availability  # 新增参数
    )
    
    fairness_results = evaluator.compute_group_metrics(
        y_true, y_score, meta_attrs, task_type
    )
    
    
    return fairness_results
