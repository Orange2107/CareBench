#!/bin/bash

# ============================================================================
# InfoReg BAYESIAN OPTIMIZATION SEARCH - MODIFY THESE VALUES TO CUSTOMIZE THE SEARCH
# ============================================================================

# Fold selection for bayesian search (can modify to include more folds)
SEARCH_FOLDS=(1)  

# Model Configuration
MODEL="inforeg"
TASK="los"  # phenotype, mortality, or los
GPU=0

# Basic Experiment Settings
PRETRAINED=true
USE_DEMOGRAPHICS=false
CROSS_EVAL=""  # Set to "matched_to_full" or "full_to_matched" if needed
MATCHED=true
INPUT_DIM=49  # EHR input dimension 

# Bayesian Optimization Settings
N_CALLS=20                    # Total number of optimization iterations
N_INITIAL_POINTS=5            # Number of random initial points
ACQUISITION_FUNC="gp_hedge"   # Acquisition function: 'LCB', 'EI', 'PI', 'gp_hedge'
N_JOBS=8                      # Number of parallel jobs (-1 for all cores)

# Resume settings
RESUME_FROM_CHECKPOINT=false  # Set to true to resume from previous run
CHECKPOINT_FILE=""            # Path to previous bayesian_optimization_result.pkl (auto-detect if empty)

# Search Space Bounds - Define parameter ranges for Bayesian optimization

# Core training parameters
LR_MIN=0.0001                # Minimum learning rate
LR_MAX=0.0001                 # Maximum learning rate
BATCH_SIZE_CHOICES="16"    # Multiple batch sizes
EPOCHS_VALUES=(50)            # Keep epochs reasonable
PATIENCE_VALUES=(15)          # Patience for early stopping

# Seeds for multiple runs
SEEDS=(42 123 1234)

# Task-specific parameters
NUM_CLASSES_VALUES=(7)       # For phenotype: 25, mortality: 1, los: 7 (auto-adjusted)

# InfoReg-specific encoder parameters
EHR_ENCODER_CHOICES="transformer"    # EHR encoder options
CXR_ENCODER_CHOICES="resnet50"       # CXR encoder options

# Architecture parameters
HIDDEN_SIZE_CHOICES="256"         # Multiple hidden sizes
EHR_DROPOUT_MIN=0.2                       # Minimum dropout for optimization
EHR_DROPOUT_MAX=0.2                       # Maximum dropout for optimization

# EHR Transformer-specific parameters (when ehr_encoder = 'transformer')
EHR_N_HEAD_CHOICES="4"                  # Number of attention heads
EHR_N_LAYERS_CHOICES="1"                # Number of transformer layers

# EHR LSTM-specific parameters (when ehr_encoder = 'lstm')
EHR_LSTM_BIDIRECTIONAL_CHOICES="true"  # Both options
EHR_LSTM_NUM_LAYERS_CHOICES="1"          # Multiple layer options

# InfoReg-specific hyperparameters (continuous ranges)
INFO_K_THRESHOLD_MIN=0.01              # Minimum k threshold
INFO_K_THRESHOLD_MAX=0.10              # Maximum k threshold
INFO_EHR_SCALE_MIN=0.5                 # Minimum EHR regularization scale
INFO_EHR_SCALE_MAX=1.5                 # Maximum EHR regularization scale
INFO_CXR_SCALE_MIN=0.05                # Minimum CXR regularization scale
INFO_CXR_SCALE_MAX=0.50                # Maximum CXR regularization scale
INFO_HISTORY_CHOICES="5,10,15"      # History length options

# ============================================================================
# SCRIPT IMPLEMENTATION - GENERALLY NO NEED TO MODIFY BELOW THIS LINE
# ============================================================================

# Function to generate dynamic results directory
generate_results_dir() {
    local model=$1
    local task=$2
    local use_demographics=$3
    local cross_eval=$4
    local matched=$5
    local pretrained=$6
    
    local demographic_str
    if [ "$use_demographics" = "true" ]; then
        demographic_str="demo"
    else
        demographic_str="no_demo"
    fi
    
    local matched_str
    if [ "$matched" = "true" ]; then
        matched_str="matched"
    else
        matched_str="full"
    fi
    
    local pretrained_str
    if [ "$pretrained" = "true" ]; then
        pretrained_str="pretrained"
    else
        pretrained_str="no_pretrained"
    fi
    
    # Handle cross_eval parameter
    local cross_eval_str
    if [ -n "$cross_eval" ]; then
        cross_eval_str="$cross_eval"
    else
        cross_eval_str="standard"
    fi
    
    # Generate results directory name
    local results_dirname="${model}_${task}-${demographic_str}-${cross_eval_str}-${matched_str}-${pretrained_str}_bayesian_search_results"
    
    echo "${BASE_DIR}/../bayesian_search_experiments/${model}/${task}/lightning_logs/${results_dirname}"
}

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR=$(generate_results_dir "$MODEL" "$TASK" "$USE_DEMOGRAPHICS" "$CROSS_EVAL" "$MATCHED" "$PRETRAINED")
LOG_FILE="${RESULTS_DIR}/bayesian_search_$(date +%Y%m%d_%H%M%S).log"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create the Bayesian optimization script
create_bayesian_optimizer() {
cat > "${RESULTS_DIR}/bayesian_optimizer.py" << 'EOF'
import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Check if skopt is available, if not install it
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
    from skopt import dump, load
    print("scikit-optimize is available")
except ImportError:
    print("Installing scikit-optimize...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize"])
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
    from skopt import dump, load

class BayesianInfoRegOptimizer:
    def __init__(self, config):
        self.config = config
        self.results_dir = config['results_dir']
        self.log_file = config['log_file']
        self.iteration = 0
        self.best_score = -np.inf
        self.best_params = None
        
        # Initialize results tracking
        self.results_data = []
        
        # Resume from checkpoint logic
        self.previous_result = None
        if config.get('resume_from_checkpoint', False):
            checkpoint_file = config.get('checkpoint_file', '')
            if not checkpoint_file:
                # Auto-detect checkpoint file
                checkpoint_file = os.path.join(self.results_dir, "bayesian_optimization_result.pkl")
            
            if os.path.exists(checkpoint_file):
                try:
                    self.previous_result = load(checkpoint_file)
                    self.log(f"Loaded checkpoint from: {checkpoint_file}")
                    self.log(f"Previous optimization had {len(self.previous_result.x_iters)} iterations")
                    
                    # Load previous results data if exists
                    csv_file = os.path.join(self.results_dir, "results_summary.csv")
                    if os.path.exists(csv_file):
                        prev_df = pd.read_csv(csv_file)
                        self.results_data = prev_df.to_dict('records')
                        self.iteration = len(self.results_data)
                        
                        # Find best previous result
                        if len(prev_df) > 0:
                            if 'acc_mean' in prev_df.columns:
                                best_row = prev_df.loc[prev_df['acc_mean'].idxmax()]
                                self.best_score = best_row['acc_mean']
                            elif 'prauc_mean' in prev_df.columns:
                                best_row = prev_df.loc[prev_df['prauc_mean'].idxmax()]
                                self.best_score = best_row['prauc_mean']
                            
                    metric_name = 'ACC' if config['task'] == 'los' else 'PRAUC'
                    self.log(f"Resuming from iteration {self.iteration}, best {metric_name} so far: {self.best_score:.4f}")
                    
                except Exception as e:
                    self.log(f"Failed to load checkpoint: {e}, starting fresh")
                    self.previous_result = None
            else:
                self.log(f"Checkpoint file not found: {checkpoint_file}, starting fresh")
        
        # Check if learning rate is fixed (min == max)
        self.fixed_lr = None
        if abs(config['lr_min'] - config['lr_max']) < 1e-10:
            self.fixed_lr = config['lr_min']
            self.log(f"Learning rate is fixed at: {self.fixed_lr}")
        
        # Check if dropout is fixed (min == max)
        self.fixed_ehr_dropout = None
        if abs(config['ehr_dropout_min'] - config['ehr_dropout_max']) < 1e-10:
            self.fixed_ehr_dropout = config['ehr_dropout_min']
            self.log(f"EHR dropout is fixed at: {self.fixed_ehr_dropout}")
        
        # Define search space for InfoReg
        self.dimensions = []
        
        # Only add lr to search space if it's not fixed
        if self.fixed_lr is None:
            self.dimensions.append(Real(config['lr_min'], config['lr_max'], name='lr'))
        
        # Add other dimensions
        dimensions_list = [
            Categorical(config['batch_size_choices'], name='batch_size'),
            Categorical(config['ehr_encoder_choices'], name='ehr_encoder'),
            Categorical(config['cxr_encoder_choices'], name='cxr_encoder'),
            Categorical(config['hidden_size_choices'], name='hidden_size'),
        ]
        
        # Only add dropout to search space if it's not fixed
        if self.fixed_ehr_dropout is None:
            dimensions_list.append(Real(config['ehr_dropout_min'], config['ehr_dropout_max'], name='ehr_dropout'))
        
        dimensions_list.extend([
            Categorical(config['ehr_n_head_choices'], name='ehr_n_head'),
            Categorical(config['ehr_n_layers_choices'], name='ehr_n_layers'),
            Categorical(config['ehr_lstm_bidirectional_choices'], name='ehr_lstm_bidirectional'),
            Categorical(config['ehr_lstm_num_layers_choices'], name='ehr_lstm_num_layers'),
            Real(config['info_k_threshold_min'], config['info_k_threshold_max'], name='info_k_threshold'),
            Real(config['info_ehr_scale_min'], config['info_ehr_scale_max'], name='info_ehr_scale'),
            Real(config['info_cxr_scale_min'], config['info_cxr_scale_max'], name='info_cxr_scale'),
            Categorical(config['info_history_choices'], name='info_history')
        ])
        
        self.dimensions.extend(dimensions_list)
        
        self.dimension_names = [dim.name for dim in self.dimensions]
        
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def is_valid_inforeg_config(self, params_dict):
        """Check if InfoReg configuration is valid"""
        # Check if transformer head configuration is valid when using transformer encoder
        if params_dict['ehr_encoder'] == 'transformer':
            hidden_size = params_dict['hidden_size']
            n_head = params_dict['ehr_n_head']
            return hidden_size % n_head == 0
        return True
    
    def extract_metrics_from_log(self, log_file):
        """Extract all performance metrics from experiment log (using overall fused results only)"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            metrics = {}
            task_type = self.config.get('task', 'phenotype')
            
            # Find the first evaluation result (overall fused result)
            # This is the result after "Testing DataLoader 0: 100%" and before the second "Evaluating {task} task..."
            
            # Find the position of "Testing DataLoader 0: 100%"
            test_complete_pos = content.find("Testing DataLoader 0: 100%")
            if test_complete_pos == -1:
                # Fallback: look for any "Testing DataLoader" completion
                test_complete_pos = content.rfind("Testing DataLoader")
            
            if test_complete_pos != -1:
                # Extract content starting from test completion
                test_section = content[test_complete_pos:]
                
                # Find the first "Evaluating {task} task..." in this section
                eval_pattern = f"Evaluating {task_type} task..."
                first_eval_pos = test_section.find(eval_pattern)
                if first_eval_pos != -1:
                    # Find the second "Evaluating {task} task..." to mark the end of first result
                    second_eval_pos = test_section.find(eval_pattern, first_eval_pos + 1)
                    
                    if second_eval_pos != -1:
                        # Extract only the first evaluation section
                        first_result_section = test_section[first_eval_pos:second_eval_pos]
                    else:
                        # If no second evaluation found, take everything after first evaluation
                        first_result_section = test_section[first_eval_pos:]
                    
                    # Extract metrics based on task type
                    if task_type == 'los':
                        # LOS task uses multiclass metrics
                        patterns = {
                            'ACC': r"overall/ACC:\s*([0-9]+\.[0-9]+)",
                            'F1_macro': r"overall/F1_macro:\s*([0-9]+\.[0-9]+)",
                            'F1_weighted': r"overall/F1_weighted:\s*([0-9]+\.[0-9]+)",
                            'Precision_macro': r"overall/Precision_macro:\s*([0-9]+\.[0-9]+)",
                            'Precision_weighted': r"overall/Precision_weighted:\s*([0-9]+\.[0-9]+)",
                            'Recall_macro': r"overall/Recall_macro:\s*([0-9]+\.[0-9]+)",
                            'Recall_weighted': r"overall/Recall_weighted:\s*([0-9]+\.[0-9]+)",
                            'Kappa': r"overall/Kappa:\s*([0-9]+\.[0-9]+)"
                        }
                    else:
                        # Phenotype/mortality tasks use multilabel/binary metrics
                        patterns = {
                            'PRAUC': r"overall/PRAUC:\s*([0-9]+\.[0-9]+)",
                            'AUROC': r"overall/AUROC:\s*([0-9]+\.[0-9]+)",
                            'ACC': r"overall/ACC:\s*([0-9]+\.[0-9]+)",
                            'F1': r"overall/F1:\s*([0-9]+\.[0-9]+)",
                            'Precision': r"overall/Precision:\s*([0-9]+\.[0-9]+)",
                            'Recall': r"overall/Recall:\s*([0-9]+\.[0-9]+)",
                            'Specificity': r"overall/Specificity:\s*([0-9]+\.[0-9]+)"
                        }
                    
                    for metric_name, pattern in patterns.items():
                        matches = re.findall(pattern, first_result_section)
                        if matches:
                            metrics[metric_name] = float(matches[0])  # Take the first (and only) match from this section
                        else:
                            metrics[metric_name] = None
                else:
                    # Fallback: use original method if pattern not found
                    if task_type == 'los':
                        patterns = {
                            'ACC': r"overall/ACC:\s*([0-9]+\.[0-9]+)",
                            'F1_macro': r"overall/F1_macro:\s*([0-9]+\.[0-9]+)",
                            'F1_weighted': r"overall/F1_weighted:\s*([0-9]+\.[0-9]+)",
                            'Precision_macro': r"overall/Precision_macro:\s*([0-9]+\.[0-9]+)",
                            'Precision_weighted': r"overall/Precision_weighted:\s*([0-9]+\.[0-9]+)",
                            'Recall_macro': r"overall/Recall_macro:\s*([0-9]+\.[0-9]+)",
                            'Recall_weighted': r"overall/Recall_weighted:\s*([0-9]+\.[0-9]+)",
                            'Kappa': r"overall/Kappa:\s*([0-9]+\.[0-9]+)"
                        }
                    else:
                        patterns = {
                            'PRAUC': r"overall/PRAUC:\s*([0-9]+\.[0-9]+)",
                            'AUROC': r"overall/AUROC:\s*([0-9]+\.[0-9]+)",
                            'ACC': r"overall/ACC:\s*([0-9]+\.[0-9]+)",
                            'F1': r"overall/F1:\s*([0-9]+\.[0-9]+)",
                            'Precision': r"overall/Precision:\s*([0-9]+\.[0-9]+)",
                            'Recall': r"overall/Recall:\s*([0-9]+\.[0-9]+)",
                            'Specificity': r"overall/Specificity:\s*([0-9]+\.[0-9]+)"
                        }
                    
                    for metric_name, pattern in patterns.items():
                        matches = re.findall(pattern, content)
                        if matches:
                            metrics[metric_name] = float(matches[0])  # Take the first match (overall fused result)
                        else:
                            metrics[metric_name] = None
            else:
                # Fallback: use original method if test completion not found
                if task_type == 'los':
                    patterns = {
                        'ACC': r"overall/ACC:\s*([0-9]+\.[0-9]+)",
                        'F1_macro': r"overall/F1_macro:\s*([0-9]+\.[0-9]+)",
                        'F1_weighted': r"overall/F1_weighted:\s*([0-9]+\.[0-9]+)",
                        'Precision_macro': r"overall/Precision_macro:\s*([0-9]+\.[0-9]+)",
                        'Precision_weighted': r"overall/Precision_weighted:\s*([0-9]+\.[0-9]+)",
                        'Recall_macro': r"overall/Recall_macro:\s*([0-9]+\.[0-9]+)",
                        'Recall_weighted': r"overall/Recall_weighted:\s*([0-9]+\.[0-9]+)",
                        'Kappa': r"overall/Kappa:\s*([0-9]+\.[0-9]+)"
                    }
                else:
                    patterns = {
                        'PRAUC': r"overall/PRAUC:\s*([0-9]+\.[0-9]+)",
                        'AUROC': r"overall/AUROC:\s*([0-9]+\.[0-9]+)",
                        'ACC': r"overall/ACC:\s*([0-9]+\.[0-9]+)",
                        'F1': r"overall/F1:\s*([0-9]+\.[0-9]+)",
                        'Precision': r"overall/Precision:\s*([0-9]+\.[0-9]+)",
                        'Recall': r"overall/Recall:\s*([0-9]+\.[0-9]+)",
                        'Specificity': r"overall/Specificity:\s*([0-9]+\.[0-9]+)"
                    }
                
                for metric_name, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    if matches:
                        metrics[metric_name] = float(matches[0])  # Take the first match (overall fused result)
                    else:
                        metrics[metric_name] = None
                    
            return metrics
        except Exception:
            return {}
    
    def run_experiment_with_seeds(self, params_dict, fold):
        """Run experiment with multiple seeds and return statistics"""
        # Check if configuration is valid for InfoReg
        if not self.is_valid_inforeg_config(params_dict):
            self.log(f"Skipping invalid config: hidden_size={params_dict['hidden_size']}, ehr_n_head={params_dict['ehr_n_head']} (not divisible)")
            return -1.0, -1.0  # Return poor scores for invalid configurations
        
        self.iteration += 1
        
        # Create experiment name
        exp_name = f"bayes_iter{self.iteration}_fold{fold}_lr{params_dict['lr']:.6f}_bs{params_dict['batch_size']}_" \
                   f"ehr_{params_dict['ehr_encoder']}_cxr_{params_dict['cxr_encoder']}_hs{params_dict['hidden_size']}_" \
                   f"edr{params_dict['ehr_dropout']:.3f}_" \
                   f"kt{params_dict['info_k_threshold']:.4f}_es{params_dict['info_ehr_scale']:.3f}_" \
                   f"cs{params_dict['info_cxr_scale']:.3f}_hist{params_dict['info_history']}"
        
        self.log(f"Starting Bayesian iteration {self.iteration}: {exp_name}")
        
        # Run experiments for all seeds
        all_metrics = []
        
        for seed in self.config['seeds']:
            seed_exp_name = f"{exp_name}_seed{seed}"
            self.log(f"  Running seed {seed}...")
            
            # Build command for InfoReg
            cmd = [
                "python", "../main.py",
                "--model", self.config['model'],
                "--mode", "train",
                "--task", self.config['task'],
                "--fold", str(fold),
                "--gpu", str(self.config['gpu']),
                "--lr", str(params_dict['lr']),
                "--batch_size", str(params_dict['batch_size']),
                "--epochs", str(self.config['epochs']),
                "--patience", str(self.config['patience']),
                "--input_dim", str(self.config['input_dim']),
                "--num_classes", str(self.config['num_classes']),
                "--ehr_encoder", params_dict['ehr_encoder'],
                "--cxr_encoder", params_dict['cxr_encoder'],
                "--hidden_size", str(params_dict['hidden_size']),
                "--ehr_dropout", str(params_dict['ehr_dropout']),
                "--ehr_n_head", str(params_dict['ehr_n_head']),
                "--ehr_num_layers", str(params_dict['ehr_n_layers']),
                "--info_k_threshold", str(params_dict['info_k_threshold']),
                "--info_ehr_scale", str(params_dict['info_ehr_scale']),
                "--info_cxr_scale", str(params_dict['info_cxr_scale']),
                "--info_history", str(params_dict['info_history']),
                "--seed", str(seed),
                "--log_dir", f"../bayesian_search_experiments/{self.config['model']}/{self.config['task']}"
            ]
            
            # Add conditional parameters (boolean flags)
            if self.config['pretrained']:
                cmd.append("--pretrained")
                
            if self.config['matched']:
                cmd.append("--matched")

            if self.config['use_demographics']:
                cmd.append("--use_demographics")

            # Handle boolean parameters from optimization
            if params_dict['ehr_lstm_bidirectional'] == 'true':
                cmd.append("--ehr_bidirectional")

            if self.config['cross_eval']:
                cmd.extend(["--cross_eval", self.config['cross_eval']])
            
            # Create experiment directory for this seed
            seed_exp_dir = os.path.join(self.results_dir, seed_exp_name)
            os.makedirs(seed_exp_dir, exist_ok=True)
            
            # Run experiment
            try:
                with open(os.path.join(seed_exp_dir, "output.log"), "w") as output_file:
                    result = subprocess.run(
                        cmd,
                        cwd=self.config['base_dir'],
                        stdout=output_file,
                        stderr=subprocess.STDOUT,
                        timeout=None
                    )
                
                # Extract all metrics
                metrics = self.extract_metrics_from_log(os.path.join(seed_exp_dir, "output.log"))
                
                if metrics and any(v is not None for v in metrics.values()):
                    all_metrics.append(metrics)
                    self.log(f"    Seed {seed}: " + " | ".join([f"{k}={v:.4f}" if v is not None else f"{k}=N/A" for k, v in metrics.items()]))
                else:
                    self.log(f"    Seed {seed}: Failed to extract metrics")
                    
            except subprocess.TimeoutExpired:
                self.log(f"    Seed {seed}: Timed out")
            except Exception as e:
                self.log(f"    Seed {seed}: Error - {e}")
        
        # Calculate statistics for all metrics
        if len(all_metrics) > 0:
            # Calculate mean and std for each metric based on task type
            task_type = self.config.get('task', 'phenotype')
            metric_stats = {}
            
            if task_type == 'los':
                # LOS task metrics
                metric_names = ['ACC', 'F1_macro', 'F1_weighted', 'Precision_macro', 'Precision_weighted', 'Recall_macro', 'Recall_weighted', 'Kappa']
            else:
                # Phenotype/mortality task metrics
                metric_names = ['PRAUC', 'AUROC', 'ACC', 'F1', 'Precision', 'Recall', 'Specificity']
            
            for metric_name in metric_names:
                values = [m.get(metric_name) for m in all_metrics if m.get(metric_name) is not None]
                if values:
                    metric_stats[f'{metric_name}_mean'] = np.mean(values)
                    metric_stats[f'{metric_name}_std'] = np.std(values)
                else:
                    metric_stats[f'{metric_name}_mean'] = None
                    metric_stats[f'{metric_name}_std'] = None
            
            # Log results
            result_str = f"Iteration {self.iteration} - "
            result_parts = []
            for metric_name in metric_names:
                mean_key = f'{metric_name}_mean'
                std_key = f'{metric_name}_std'
                if metric_stats[mean_key] is not None:
                    result_parts.append(f"{metric_name}: {metric_stats[mean_key]:.4f}±{metric_stats[std_key]:.4f}")
                else:
                    result_parts.append(f"{metric_name}: N/A")
            
            self.log(result_str + " | ".join(result_parts))
            
            # Update best result based on task type
            if task_type == 'los':
                # Use ACC for LOS task
                acc_mean = metric_stats.get('ACC_mean')
                if acc_mean is not None and acc_mean > self.best_score:
                    self.best_score = acc_mean
                    self.best_params = params_dict.copy()
                    self.log(f"New best ACC: {self.best_score:.4f}±{metric_stats.get('ACC_std', 0):.4f}")
                
                # Save result
                result_data = {
                    'iteration': self.iteration,
                    'experiment_name': exp_name,
                    'fold': fold,
                    **params_dict,
                    'task': self.config['task'],
                    'use_demographics': self.config['use_demographics'],
                    'cross_eval': self.config['cross_eval'],
                    'pretrained': self.config['pretrained'],
                    **metric_stats,
                    'all_metrics': all_metrics
                }
                self.results_data.append(result_data)
                
                return acc_mean if acc_mean is not None else -1.0, metric_stats.get('F1_macro_mean', -1.0)
            else:
                # Use PRAUC for phenotype/mortality tasks
                prauc_mean = metric_stats.get('PRAUC_mean')
                if prauc_mean is not None and prauc_mean > self.best_score:
                    self.best_score = prauc_mean
                    self.best_params = params_dict.copy()
                    self.log(f"New best PRAUC: {self.best_score:.4f}±{metric_stats.get('PRAUC_std', 0):.4f}")
                
                # Save result
                result_data = {
                    'iteration': self.iteration,
                    'experiment_name': exp_name,
                    'fold': fold,
                    **params_dict,
                    'task': self.config['task'],
                    'use_demographics': self.config['use_demographics'],
                    'cross_eval': self.config['cross_eval'],
                    'pretrained': self.config['pretrained'],
                    **metric_stats,
                    'all_metrics': all_metrics
                }
                self.results_data.append(result_data)
                
                return prauc_mean if prauc_mean is not None else -1.0, metric_stats.get('AUROC_mean', -1.0)
        else:
            self.log(f"Failed to get valid results from any seed in iteration {self.iteration}")
            return -1.0, -1.0
    
    def objective_function(self, params):
        """Objective function for Bayesian optimization"""
        # Convert params list to dict
        params_dict = dict(zip(self.dimension_names, params))
        
        # Add fixed learning rate if it was not in search space
        if self.fixed_lr is not None:
            params_dict['lr'] = self.fixed_lr
        
        # Add fixed dropout if it was not in search space
        if self.fixed_ehr_dropout is not None:
            params_dict['ehr_dropout'] = self.fixed_ehr_dropout
        
        # Validate InfoReg configuration first
        if not self.is_valid_inforeg_config(params_dict):
            # Return very poor score for invalid configurations
            return 1.0  # High value because we minimize
        
        # Run experiments for all folds and average the results
        scores = []
        for fold in self.config['search_folds']:
            score, _ = self.run_experiment_with_seeds(params_dict, fold)
            scores.append(score)
        
        # Return negative score because skopt minimizes
        avg_score = np.mean(scores)
        task_type = self.config.get('task', 'phenotype')
        metric_name = 'ACC' if task_type == 'los' else 'PRAUC'
        return -avg_score  # Negative because we want to maximize ACC (LOS) or PRAUC (phenotype)
    
    def run_optimization(self):
        """Run Bayesian optimization"""
        self.log("Starting Bayesian Optimization for InfoReg")
        self.log(f"Seeds: {self.config['seeds']}")
        if self.fixed_lr is not None:
            self.log(f"Fixed parameter: lr={self.fixed_lr}")
        if self.fixed_ehr_dropout is not None:
            self.log(f"Fixed parameter: ehr_dropout={self.fixed_ehr_dropout}")
        self.log(f"Search space: {[dim.name for dim in self.dimensions]}")
        
        if self.previous_result:
            # Continue from previous optimization
            remaining_calls = self.config['n_calls'] - len(self.previous_result.x_iters)
            if remaining_calls <= 0:
                self.log("Previous optimization already completed the requested iterations")
                return self.previous_result
                
            self.log(f"Continuing optimization: {remaining_calls} remaining calls")
            
            # Continue optimization using tell/ask interface
            from skopt import Optimizer
            
            # Create optimizer with same settings
            opt = Optimizer(
                dimensions=self.dimensions,
                acq_func=self.config['acquisition_func'],
                n_initial_points=0,  # No initial points needed
                random_state=42
            )
            
            # Tell the optimizer about previous results
            for x, y in zip(self.previous_result.x_iters, self.previous_result.func_vals):
                opt.tell(x, y)
                
            # Continue optimization
            for i in range(remaining_calls):
                next_x = opt.ask()
                next_y = self.objective_function(next_x)
                opt.tell(next_x, next_y)
                
                # Save checkpoint periodically
                if (i + 1) % 5 == 0:  # Save every 5 iterations
                    optimization_file = os.path.join(self.results_dir, "bayesian_optimization_result.pkl")
                    dump(opt, optimization_file)
                    self.log(f"Checkpoint saved at iteration {len(self.previous_result.x_iters) + i + 1}")
            
            result = opt
            
        else:
            # Fresh start
            self.log(f"Total iterations: {self.config['n_calls']}")
            self.log(f"Initial random points: {self.config['n_initial_points']}")
            
            # Run optimization
            result = gp_minimize(
                func=self.objective_function,
                dimensions=self.dimensions,
                n_calls=self.config['n_calls'],
                n_initial_points=self.config['n_initial_points'],
                acq_func=self.config['acquisition_func'],
                n_jobs=self.config['n_jobs'],
                random_state=42
            )
        
        # Save optimization result
        optimization_file = os.path.join(self.results_dir, "bayesian_optimization_result.pkl")
        dump(result, optimization_file)
        
        # Save all results to CSV
        if self.results_data:
            df = pd.DataFrame(self.results_data)
            csv_file = os.path.join(self.results_dir, "results_summary.csv")
            df.to_csv(csv_file, index=False)
        
        # Final analysis
        task_type = self.config.get('task', 'phenotype')
        metric_name = 'ACC' if task_type == 'los' else 'PRAUC'
        self.log("=== BAYESIAN OPTIMIZATION COMPLETED ===")
        self.log(f"Best {metric_name} found: {self.best_score:.4f}")
        self.log(f"Best parameters: {self.best_params}")
        
        # Save best parameters
        best_params_file = os.path.join(self.results_dir, "best_params.txt")
        with open(best_params_file, 'w') as f:
            f.write("InfoReg Bayesian Optimization Best Parameters\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best {metric_name}: {self.best_score:.4f}\n")
            f.write(f"Total iterations: {self.iteration}\n\n")
            f.write("Best Parameters:\n")
            if self.best_params:
                for param, value in self.best_params.items():
                    f.write(f"  {param}: {value}\n")
            f.write(f"Seeds used: {self.config['seeds']}\n")
        
        # Generate convergence plot
        self.generate_convergence_plot(result)
        
        return result
    
    def generate_convergence_plot(self, result):
        """Generate convergence plot"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10), facecolor='white')
            
            # Plot convergence
            task_type = self.config.get('task', 'phenotype')
            metric_name = 'ACC' if task_type == 'los' else 'PRAUC'
            scores = [-y for y in result.func_vals]  # Convert back to positive
            best_scores = [max(scores[:i+1]) for i in range(len(scores))]
            
            plt.subplot(2, 3, 1)
            plt.plot(scores, 'bo-', alpha=0.6, label=metric_name)
            plt.plot(best_scores, 'r-', linewidth=2, label=f'Best {metric_name}')
            plt.xlabel('Iteration')
            plt.ylabel(metric_name)
            plt.title('Bayesian Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot parameter exploration
            if hasattr(result, 'x_iters') and len(result.x_iters) > 5:
                param_data = pd.DataFrame(result.x_iters, columns=self.dimension_names)
                
                # Learning rate vs Performance (only if lr is in search space)
                if 'lr' in self.dimension_names:
                    plt.subplot(2, 3, 2)
                    plt.scatter(param_data['lr'], scores, alpha=0.6, c=scores, cmap='viridis')
                    plt.xlabel('Learning Rate')
                    plt.ylabel(metric_name)
                    plt.title('Learning Rate vs Performance')
                    plt.colorbar(label=metric_name)
                    plt.grid(True, alpha=0.3)
                else:
                    # If lr is fixed, show fixed value
                    plt.subplot(2, 3, 2)
                    fixed_params_text = f'LR fixed at {self.fixed_lr}'
                    if self.fixed_ehr_dropout is not None:
                        fixed_params_text += f'\nDropout fixed at {self.fixed_ehr_dropout}'
                    plt.text(0.5, 0.5, fixed_params_text, 
                            ha='center', va='center', fontsize=11, 
                            transform=plt.gca().transAxes)
                    plt.title('Fixed Parameters')
                    plt.axis('off')
                
                # InfoReg parameters exploration
                plt.subplot(2, 3, 3)
                plt.scatter(param_data['info_ehr_scale'], param_data['info_cxr_scale'], 
                           c=scores, cmap='viridis', alpha=0.7)
                plt.colorbar(label=metric_name)
                plt.xlabel('EHR Scale')
                plt.ylabel('CXR Scale')
                plt.title('InfoReg Scales vs Performance')
                plt.grid(True, alpha=0.3)
                
                # Encoder combination exploration
                plt.subplot(2, 3, 4)
                ehr_encoders = param_data['ehr_encoder'].unique()
                cxr_encoders = param_data['cxr_encoder'].unique()
                
                for ehr_enc in ehr_encoders:
                    for cxr_enc in cxr_encoders:
                        mask = (param_data['ehr_encoder'] == ehr_enc) & (param_data['cxr_encoder'] == cxr_enc)
                        if mask.any():
                            combo_scores = [scores[k] for k in range(len(scores)) if mask.iloc[k]]
                            plt.scatter([f"{ehr_enc}+{cxr_enc}"] * len(combo_scores), combo_scores, 
                                      alpha=0.7, label=f"{ehr_enc}+{cxr_enc}")
                plt.xlabel('Encoder Combination')
                plt.ylabel(metric_name)
                plt.title('Performance by Encoder Combination')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # K threshold vs performance
                plt.subplot(2, 3, 5)
                plt.scatter(param_data['info_k_threshold'], scores, 
                           alpha=0.7, c=scores, cmap='viridis')
                plt.colorbar(label=metric_name)
                plt.xlabel('K Threshold')
                plt.ylabel(metric_name)
                plt.title('K Threshold vs Performance')
                plt.grid(True, alpha=0.3)
                
                # History length vs performance
                plt.subplot(2, 3, 6)
                for history in param_data['info_history'].unique():
                    mask = param_data['info_history'] == history
                    hist_scores = [scores[i] for i in range(len(scores)) if mask.iloc[i]]
                    plt.scatter([history] * len(hist_scores), hist_scores, alpha=0.7, label=f"History={history}")
                plt.xlabel('History Length')
                plt.ylabel(metric_name)
                plt.title('History Length vs Performance')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, "convergence_plot.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log(f"Convergence plot saved to: {plot_file}")
            
        except Exception as e:
            self.log(f"Could not generate convergence plot: {e}")

def main():
    # Read configuration from environment variables
    config = {
        'results_dir': os.environ.get('RESULTS_DIR'),
        'log_file': os.environ.get('LOG_FILE'),
        'base_dir': os.environ.get('BASE_DIR'),
        'model': os.environ.get('MODEL'),
        'task': os.environ.get('TASK'),
        'gpu': int(os.environ.get('GPU', 0)),
        'epochs': int(os.environ.get('EPOCHS', 50)),
        'patience': int(os.environ.get('PATIENCE', 10)),
        'input_dim': int(os.environ.get('INPUT_DIM', 49)),
        'num_classes': int(os.environ.get('NUM_CLASSES', 25)),
        'pretrained': os.environ.get('PRETRAINED', 'true').lower() == 'true',
        'matched': os.environ.get('MATCHED', 'true').lower() == 'true',
        'use_demographics': os.environ.get('USE_DEMOGRAPHICS', 'false').lower() == 'true',
        'cross_eval': os.environ.get('CROSS_EVAL', ''),
        'search_folds': [int(x) for x in os.environ.get('SEARCH_FOLDS', '1').split(',')],
        
        # Bayesian optimization parameters
        'n_calls': int(os.environ.get('N_CALLS', 20)),
        'n_initial_points': int(os.environ.get('N_INITIAL_POINTS', 5)),
        'acquisition_func': os.environ.get('ACQUISITION_FUNC', 'gp_hedge'),
        'n_jobs': int(os.environ.get('N_JOBS', 1)),
        
        # Resume parameters
        'resume_from_checkpoint': os.environ.get('RESUME_FROM_CHECKPOINT', 'false').lower() == 'true',
        'checkpoint_file': os.environ.get('CHECKPOINT_FILE', ''),
        
        # Search space bounds
        'lr_min': float(os.environ.get('LR_MIN', 0.00005)),
        'lr_max': float(os.environ.get('LR_MAX', 0.0005)),
        'seeds': [int(x) for x in os.environ.get('SEEDS', '42,123,1234').split(',')],
        'batch_size_choices': [int(x) for x in os.environ.get('BATCH_SIZE_CHOICES', '16,32').split(',')],
        'ehr_dropout_min': float(os.environ.get('EHR_DROPOUT_MIN', 0.1)),
        'ehr_dropout_max': float(os.environ.get('EHR_DROPOUT_MAX', 0.5)),
        
        # InfoReg-specific parameters
        'ehr_encoder_choices': os.environ.get('EHR_ENCODER_CHOICES', 'transformer,lstm').split(','),
        'cxr_encoder_choices': os.environ.get('CXR_ENCODER_CHOICES', 'resnet50').split(','),
        'hidden_size_choices': [int(x) for x in os.environ.get('HIDDEN_SIZE_CHOICES', '128,256,512').split(',')],
        'ehr_n_head_choices': [int(x) for x in os.environ.get('EHR_N_HEAD_CHOICES', '4,8').split(',')],
        'ehr_n_layers_choices': [int(x) for x in os.environ.get('EHR_N_LAYERS_CHOICES', '1,2').split(',')],
        'ehr_lstm_bidirectional_choices': os.environ.get('EHR_LSTM_BIDIRECTIONAL_CHOICES', 'true,false').split(','),
        'ehr_lstm_num_layers_choices': [int(x) for x in os.environ.get('EHR_LSTM_NUM_LAYERS_CHOICES', '1,2,3').split(',')],
        'info_k_threshold_min': float(os.environ.get('INFO_K_THRESHOLD_MIN', 0.01)),
        'info_k_threshold_max': float(os.environ.get('INFO_K_THRESHOLD_MAX', 0.10)),
        'info_ehr_scale_min': float(os.environ.get('INFO_EHR_SCALE_MIN', 0.5)),
        'info_ehr_scale_max': float(os.environ.get('INFO_EHR_SCALE_MAX', 1.5)),
        'info_cxr_scale_min': float(os.environ.get('INFO_CXR_SCALE_MIN', 0.05)),
        'info_cxr_scale_max': float(os.environ.get('INFO_CXR_SCALE_MAX', 0.50)),
        'info_history_choices': [int(x) for x in os.environ.get('INFO_HISTORY_CHOICES', '5,10,15,20').split(',')]
    }
    
    # Auto-adjust num_classes for mortality task
    if config['task'] == 'mortality':
        config['num_classes'] = 1
    elif config['task'] == 'los':
        config['num_classes'] = 7
    
    # Create and run optimizer
    optimizer = BayesianInfoRegOptimizer(config)
    result = optimizer.run_optimization()
    
    print("InfoReg Bayesian optimization completed successfully!")

if __name__ == "__main__":
    main()
EOF
}

# Main execution
main() {
    log "Starting InfoReg Bayesian Optimization Search"
    log "Configuration: MODEL=$MODEL, TASK=$TASK, USE_DEMOGRAPHICS=$USE_DEMOGRAPHICS, CROSS_EVAL=$CROSS_EVAL, PRETRAINED=$PRETRAINED"
    log "Results will be saved to: $RESULTS_DIR"
    log "Log file: $LOG_FILE"
    log "Total optimization calls: $N_CALLS"
    log "Initial random points: $N_INITIAL_POINTS"
    log "Acquisition function: $ACQUISITION_FUNC"
    
    # Create the Bayesian optimizer Python script
    create_bayesian_optimizer
    
    # Set environment variables for the Python script
    export RESULTS_DIR="$RESULTS_DIR"
    export LOG_FILE="$LOG_FILE"
    export BASE_DIR="$BASE_DIR"
    export MODEL="$MODEL"
    export TASK="$TASK"
    export GPU="$GPU"
    export EPOCHS="${EPOCHS_VALUES[0]}"
    export PATIENCE="${PATIENCE_VALUES[0]}"
    export INPUT_DIM="$INPUT_DIM"
    export NUM_CLASSES="${NUM_CLASSES_VALUES[0]}"
    export PRETRAINED="$PRETRAINED"
    export MATCHED="$MATCHED"
    export USE_DEMOGRAPHICS="$USE_DEMOGRAPHICS"
    export CROSS_EVAL="$CROSS_EVAL"
    export SEARCH_FOLDS=$(IFS=,; echo "${SEARCH_FOLDS[*]}")
    
    # Bayesian optimization parameters
    export N_CALLS="$N_CALLS"
    export N_INITIAL_POINTS="$N_INITIAL_POINTS"
    export ACQUISITION_FUNC="$ACQUISITION_FUNC"
    export N_JOBS="$N_JOBS"
    
    # Resume parameters
    export RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT"
    export CHECKPOINT_FILE="$CHECKPOINT_FILE"
    
    # Search space bounds
    export LR_MIN="$LR_MIN"
    export LR_MAX="$LR_MAX"
    export SEEDS=$(IFS=,; echo "${SEEDS[*]}")
    export BATCH_SIZE_CHOICES="$BATCH_SIZE_CHOICES"
    export EHR_DROPOUT_MIN="$EHR_DROPOUT_MIN"
    export EHR_DROPOUT_MAX="$EHR_DROPOUT_MAX"
    
    # InfoReg-specific parameters
    export EHR_ENCODER_CHOICES="$EHR_ENCODER_CHOICES"
    export CXR_ENCODER_CHOICES="$CXR_ENCODER_CHOICES"
    export HIDDEN_SIZE_CHOICES="$HIDDEN_SIZE_CHOICES"
    export EHR_N_HEAD_CHOICES="$EHR_N_HEAD_CHOICES"
    export EHR_N_LAYERS_CHOICES="$EHR_N_LAYERS_CHOICES"
    export EHR_LSTM_BIDIRECTIONAL_CHOICES="$EHR_LSTM_BIDIRECTIONAL_CHOICES"
    export EHR_LSTM_NUM_LAYERS_CHOICES="$EHR_LSTM_NUM_LAYERS_CHOICES"
    export INFO_K_THRESHOLD_MIN="$INFO_K_THRESHOLD_MIN"
    export INFO_K_THRESHOLD_MAX="$INFO_K_THRESHOLD_MAX"
    export INFO_EHR_SCALE_MIN="$INFO_EHR_SCALE_MIN"
    export INFO_EHR_SCALE_MAX="$INFO_EHR_SCALE_MAX"
    export INFO_CXR_SCALE_MIN="$INFO_CXR_SCALE_MIN"
    export INFO_CXR_SCALE_MAX="$INFO_CXR_SCALE_MAX"
    export INFO_HISTORY_CHOICES="$INFO_HISTORY_CHOICES"
    
    log "Starting Python Bayesian optimizer..."
    
    # Run the Bayesian optimizer
    cd "$BASE_DIR"
    python3 "${RESULTS_DIR}/bayesian_optimizer.py"
    
    if [ $? -eq 0 ]; then
        log "InfoReg Bayesian optimization completed successfully!"
        log "Results saved to: $RESULTS_DIR"
        log "Best parameters in: $RESULTS_DIR/best_params.txt"
        log "Full results in: $RESULTS_DIR/results_summary.csv"
        log "Optimization object saved in: $RESULTS_DIR/bayesian_optimization_result.pkl"
    else
        log "InfoReg Bayesian optimization failed!"
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    log "InfoReg Bayesian search interrupted by user"
    exit 1
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"


