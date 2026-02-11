#!/bin/bash

# ============================================================================
# SMIL BAYESIAN OPTIMIZATION SEARCH - FINETUNE PARAMETERS ONLY
# ============================================================================

# Fold selection for bayesian search (can modify to include more folds)
SEARCH_FOLDS=(1)  

# Model Configuration
MODEL="smil"
TASK="los"  # phenotype, mortality, los
GPU=5

# Basic Experiment Settings
PRETRAINED=true
USE_DEMOGRAPHICS=false
CROSS_EVAL=""  # Set to "matched_to_full" or "full_to_matched" if needed
MATCHED=true

# Bayesian Optimization Settings
N_CALLS=20                    # Total number of optimization iterations
N_INITIAL_POINTS=5           # Number of random initial points
ACQUISITION_FUNC="gp_hedge"  # Acquisition function: 'LCB', 'EI', 'PI', 'gp_hedge'
N_JOBS=8                     # Number of parallel jobs (-1 for all cores)

# Resume settings
RESUME_FROM_CHECKPOINT=false  # Set to true to resume from previous run
CHECKPOINT_FILE=""           # Path to previous bayesian_optimization_result.pkl (auto-detect if empty)

# Search Space Bounds - Define parameter ranges for Bayesian optimization
# Format: [min_value, max_value] for continuous parameters
# For discrete parameters, we'll use choice spaces in the Python script

# Fixed parameters (not optimized)
LR_FIXED=0.0001               # Fixed learning rate
BATCH_SIZE_FIXED=16          # Fixed batch size
EPOCHS_VALUES=(50)
PATIENCE_VALUES=(10)
DROPOUT_FIXED=0.2            # Fixed dropout
HIDDEN_DIM_FIXED=256         # Fixed hidden dimension

# Seeds for multiple runs
SEEDS=(42 123 1234)

# Task-specific parameters
INPUT_DIM_VALUES=(49)         # EHR input dimension for SMIL
NUM_CLASSES_VALUES=(25)       # For phenotype task: 25, for mortality: 1, for los: 7 (auto-adjusted)

# Fixed encoder parameters (not optimized)
EHR_ENCODER_FIXED="transformer"  # Fixed EHR encoder
CXR_ENCODER_FIXED="resnet50"     # Fixed CXR encoder
EHR_N_HEAD_FIXED=4               # Fixed attention heads
EHR_N_LAYERS_FIXED=1             # Fixed transformer layers
MAX_LEN_FIXED=500                # Fixed max sequence length
N_CLUSTERS_FIXED=10              # Fixed number of clusters

# FINETUNE PARAMETERS TO OPTIMIZE (based on smil.yaml comments)
INNER_LOOP_CHOICES="1,2,3"       # Meta-learning inner loop iterations
MC_SIZE_CHOICES="10,20,30"       # Monte Carlo size
LR_INNER_MIN=0.0001               # Inner learning rate range
LR_INNER_MAX=0.001
ALPHA_MIN=0.05                   # Feature distillation weight range
ALPHA_MAX=0.2
BETA_MIN=0.05                    # EHR mean distillation weight range  
BETA_MAX=0.2
TEMPERATURE_MIN=1.0              # Knowledge distillation temperature range
TEMPERATURE_MAX=3.0

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

class BayesianSMILOptimizer:
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
                            best_row = prev_df.loc[prev_df['acc'].idxmax()]  # 改为acc
                            self.best_score = best_row['acc']
                            
                    self.log(f"Resuming from iteration {self.iteration}, best ACC so far: {self.best_score:.4f}")
                    
                except Exception as e:
                    self.log(f"Failed to load checkpoint: {e}, starting fresh")
                    self.previous_result = None
            else:
                self.log(f"Checkpoint file not found: {checkpoint_file}, starting fresh")
        
        # Define search space for SMIL - ONLY FINETUNE PARAMETERS
        self.dimensions = [
            Categorical(config['inner_loop_choices'], name='inner_loop'),
            Categorical(config['mc_size_choices'], name='mc_size'),
            Real(config['lr_inner_min'], config['lr_inner_max'], name='lr_inner', prior='log-uniform'),
            Real(config['alpha_min'], config['alpha_max'], name='alpha'),
            Real(config['beta_min'], config['beta_max'], name='beta'),
            Real(config['temperature_min'], config['temperature_max'], name='temperature')
        ]
        
        self.dimension_names = [dim.name for dim in self.dimensions]
        
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def extract_metrics_from_log(self, log_file):
        """Extract all performance metrics from experiment log"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            metrics = {}
            
            # Extract all metrics using regex patterns
            patterns = {
                'ACC': r"overall/ACC:\s*([0-9]+\.[0-9]+)",
                'F1_macro': r"overall/F1_macro:\s*([0-9]+\.[0-9]+)",
                'F1_weighted': r"overall/F1_weighted:\s*([0-9]+\.[0-9]+)",
                'Precision_macro': r"overall/Precision_macro:\s*([0-9]+\.[0-9]+)",
                'Precision_weighted': r"overall/Precision_weighted:\s*([0-9]+\.[0-9]+)",
                'Recall_macro': r"overall/Recall_macro:\s*([0-9]+\.[0-9]+)",
                'Recall_weighted': r"overall/Recall_weighted:\s*([0-9]+\.[0-9]+)"
            }
            
            for metric_name, pattern in patterns.items():
                matches = re.findall(pattern, content)
                if matches:
                    metrics[metric_name] = float(matches[-1])  # Take the last match
                else:
                    metrics[metric_name] = None
                    
            return metrics
        except Exception:
            return {}
    
    def run_experiment_with_seeds(self, params_dict, fold):
        """Run experiment with multiple seeds and return statistics"""
        self.iteration += 1
        
        # Create experiment name
        exp_name = f"bayes_iter{self.iteration}_fold{fold}_inner{params_dict['inner_loop']}_" \
                   f"mc{params_dict['mc_size']}_lrin{params_dict['lr_inner']:.4f}_" \
                   f"alpha{params_dict['alpha']:.3f}_beta{params_dict['beta']:.3f}_" \
                   f"temp{params_dict['temperature']:.1f}"
        
        self.log(f"Starting Bayesian iteration {self.iteration}: {exp_name}")
        
        # Run experiments for all seeds
        all_metrics = []
        
        for seed in self.config['seeds']:
            seed_exp_name = f"{exp_name}_seed{seed}"
            self.log(f"  Running seed {seed}...")
            
            # Build command for SMIL
            cmd = [
                "python", "../main.py",
                "--model", self.config['model'],
                "--mode", "train",
                "--task", self.config['task'],
                "--fold", str(fold),
                "--gpu", str(self.config['gpu']),
                "--lr", str(self.config['lr_fixed']),
                "--batch_size", str(self.config['batch_size_fixed']),
                "--epochs", str(self.config['epochs']),
                "--patience", str(self.config['patience']),
                "--dropout", str(self.config['dropout_fixed']),
                "--hidden_dim", str(self.config['hidden_dim_fixed']),
                "--input_dim", str(self.config['input_dim']),
                "--num_classes", str(self.config['num_classes']),
                "--ehr_encoder", self.config['ehr_encoder_fixed'],
                "--cxr_encoder", self.config['cxr_encoder_fixed'],
                "--ehr_n_head", str(self.config['ehr_n_head_fixed']),
                "--ehr_n_layers", str(self.config['ehr_n_layers_fixed']),
                "--max_len", str(self.config['max_len_fixed']),
                "--n_clusters", str(self.config['n_clusters_fixed']),
                "--seed", str(seed),
                # FINETUNE PARAMETERS
                "--inner_loop", str(params_dict['inner_loop']),
                "--lr_inner", str(params_dict['lr_inner']),
                "--mc_size", str(params_dict['mc_size']),
                "--alpha", str(params_dict['alpha']),
                "--beta", str(params_dict['beta']),
                "--temperature", str(params_dict['temperature']),
                "--log_dir", f"../bayesian_search_experiments/{self.config['model']}/{self.config['task']}"
            ]
            
            # Add conditional parameters (boolean flags)
            if self.config['pretrained']:
                cmd.append("--pretrained")
                
            if self.config['matched']:
                cmd.append("--matched")

            if self.config['use_demographics']:
                cmd.append("--use_demographics")

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
                        timeout=None  # No timeout
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
            # Calculate mean and std for each metric
            metric_stats = {}
            for metric_name in ['ACC', 'F1_macro', 'F1_weighted', 'Precision_macro', 'Precision_weighted', 'Recall_macro', 'Recall_weighted']:
                values = [m.get(metric_name) for m in all_metrics if m.get(metric_name) is not None]
                if values:
                    metric_stats[f'{metric_name}_mean'] = np.mean(values)
                    metric_stats[f'{metric_name}_std'] = np.std(values)
                else:
                    metric_stats[f'{metric_name}_mean'] = None
                    metric_stats[f'{metric_name}_std'] = None
            
            # Log results in the requested format for all metrics
            result_str = f"Iteration {self.iteration} - "
            result_parts = []
            for metric_name in ['ACC', 'F1_macro', 'F1_weighted', 'Precision_macro', 'Precision_weighted', 'Recall_macro', 'Recall_weighted']:
                mean_key = f'{metric_name}_mean'
                std_key = f'{metric_name}_std'
                if metric_stats[mean_key] is not None:
                    result_parts.append(f"{metric_name}: {metric_stats[mean_key]:.4f}±{metric_stats[std_key]:.4f}")
                else:
                    result_parts.append(f"{metric_name}: N/A")
            
            self.log(result_str + " | ".join(result_parts))
            
            # Update best result based on ACC (instead of PRAUC for LoS)
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
                **metric_stats,  # Include all metric statistics
                'all_metrics': all_metrics  # Include raw metrics from all seeds
            }
            self.results_data.append(result_data)
            
            return acc_mean if acc_mean is not None else -1.0, metric_stats.get('F1_macro_mean', -1.0)
        else:
            self.log(f"Failed to get valid results from any seed in iteration {self.iteration}")
            return -1.0, -1.0
    
    def objective_function(self, params):
        """Objective function for Bayesian optimization"""
        # Convert params list to dict
        params_dict = dict(zip(self.dimension_names, params))
        
        # Run experiments for all folds and average the results
        scores = []
        for fold in self.config['search_folds']:
            score, _ = self.run_experiment_with_seeds(params_dict, fold)
            scores.append(score)
        
        # Return negative score because skopt minimizes
        avg_score = np.mean(scores)
        return -avg_score  # Negative because we want to maximize ACC
    
    def run_optimization(self):
        """Run Bayesian optimization"""
        self.log("Starting Bayesian Optimization for SMIL (Finetune Parameters Only)")
        self.log(f"Fixed parameters: lr={self.config['lr_fixed']}, batch_size={self.config['batch_size_fixed']}, dropout={self.config['dropout_fixed']}")
        self.log(f"Fixed encoder: ehr_encoder={self.config['ehr_encoder_fixed']}, cxr_encoder={self.config['cxr_encoder_fixed']}")
        self.log(f"Seeds: {self.config['seeds']}")
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
        self.log("=== BAYESIAN OPTIMIZATION COMPLETED ===")
        self.log(f"Best ACC found: {self.best_score:.4f}")
        self.log(f"Best parameters: {self.best_params}")
        
        # Save best parameters
        best_params_file = os.path.join(self.results_dir, "best_params.txt")
        with open(best_params_file, 'w') as f:
            f.write("SMIL Bayesian Optimization Best Parameters (Finetune Only)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Best ACC: {self.best_score:.4f}\n")
            f.write(f"Total iterations: {self.iteration}\n\n")
            f.write("Best Finetune Parameters:\n")
            if self.best_params:
                for param, value in self.best_params.items():
                    f.write(f"  {param}: {value}\n")
            f.write(f"\nFixed Parameters:\n")
            f.write(f"  lr: {self.config['lr_fixed']}\n")
            f.write(f"  batch_size: {self.config['batch_size_fixed']}\n")
            f.write(f"  dropout: {self.config['dropout_fixed']}\n")
            f.write(f"  hidden_dim: {self.config['hidden_dim_fixed']}\n")
            f.write(f"  ehr_encoder: {self.config['ehr_encoder_fixed']}\n")
            f.write(f"  cxr_encoder: {self.config['cxr_encoder_fixed']}\n")
            f.write(f"  Seeds used: {self.config['seeds']}\n")
        
        # Generate convergence plot
        self.generate_convergence_plot(result)
        
        return result
    
    def generate_convergence_plot(self, result):
        """Generate convergence plot"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10), facecolor='white')
            
            # Plot convergence
            scores = [-y for y in result.func_vals]  # Convert back to positive
            best_scores = [max(scores[:i+1]) for i in range(len(scores))]
            
            plt.subplot(2, 3, 1)
            plt.plot(scores, 'bo-', alpha=0.6, label='ACC')
            plt.plot(best_scores, 'r-', linewidth=2, label='Best ACC')
            plt.xlabel('Iteration')
            plt.ylabel('ACC')
            plt.title('Bayesian Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot parameter exploration
            if hasattr(result, 'x_iters') and len(result.x_iters) > 5:
                param_data = pd.DataFrame(result.x_iters, columns=self.dimension_names)
                
                # Inner loop vs MC size
                plt.subplot(2, 3, 2)
                plt.scatter(param_data['inner_loop'], param_data['mc_size'], 
                           c=scores, cmap='viridis', alpha=0.7, s=60)
                plt.colorbar(label='ACC')
                plt.xlabel('Inner Loop')
                plt.ylabel('MC Size')
                plt.title('Inner Loop vs MC Size')
                plt.grid(True, alpha=0.3)
                
                # Learning rate inner progression
                plt.subplot(2, 3, 3)
                plt.plot(param_data['lr_inner'], 'g-', alpha=0.7, marker='o')
                plt.xlabel('Iteration')
                plt.ylabel('Inner Learning Rate')
                plt.title('Inner LR Exploration')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                
                # Alpha vs Beta relationship
                plt.subplot(2, 3, 4)
                plt.scatter(param_data['alpha'], param_data['beta'], 
                           c=scores, cmap='viridis', alpha=0.7, s=60)
                plt.colorbar(label='ACC')
                plt.xlabel('Alpha (Feature Distillation)')
                plt.ylabel('Beta (EHR Mean Distillation)')
                plt.title('Alpha vs Beta')
                plt.grid(True, alpha=0.3)
                
                # Temperature progression
                plt.subplot(2, 3, 5)
                plt.plot(param_data['temperature'], 'r-', alpha=0.7, marker='o')
                plt.xlabel('Iteration')
                plt.ylabel('Temperature')
                plt.title('Temperature Exploration')
                plt.grid(True, alpha=0.3)
                
                # Parameter correlation with performance
                plt.subplot(2, 3, 6)
                # Create correlation matrix between parameters and scores
                corr_data = param_data.copy()
                corr_data['ACC'] = scores
                
                # Select numeric columns only
                numeric_cols = ['lr_inner', 'alpha', 'beta', 'temperature', 'ACC']
                corr_matrix = corr_data[numeric_cols].corr()['ACC'].drop('ACC')
                
                plt.barh(range(len(corr_matrix)), corr_matrix.values)
                plt.yticks(range(len(corr_matrix)), corr_matrix.index)
                plt.xlabel('Correlation with ACC')
                plt.title('Parameter Importance')
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
        'input_dim': int(os.environ.get('INPUT_DIM', 24)),
        'num_classes': int(os.environ.get('NUM_CLASSES', 25)),
        'pretrained': os.environ.get('PRETRAINED', 'true').lower() == 'true',
        'matched': os.environ.get('MATCHED', 'true').lower() == 'true',
        'use_demographics': os.environ.get('USE_DEMOGRAPHICS', 'false').lower() == 'true',
        'cross_eval': os.environ.get('CROSS_EVAL', ''),
        'search_folds': [int(x) for x in os.environ.get('SEARCH_FOLDS', '1').split(',')],
        
        # Bayesian optimization parameters
        'n_calls': int(os.environ.get('N_CALLS', 30)),
        'n_initial_points': int(os.environ.get('N_INITIAL_POINTS', 8)),
        'acquisition_func': os.environ.get('ACQUISITION_FUNC', 'gp_hedge'),
        'n_jobs': int(os.environ.get('N_JOBS', 1)),
        
        # Resume parameters
        'resume_from_checkpoint': os.environ.get('RESUME_FROM_CHECKPOINT', 'false').lower() == 'true',
        'checkpoint_file': os.environ.get('CHECKPOINT_FILE', ''),
        
        # Fixed parameters
        'lr_fixed': float(os.environ.get('LR_FIXED', 0.001)),
        'batch_size_fixed': int(os.environ.get('BATCH_SIZE_FIXED', 32)),
        'dropout_fixed': float(os.environ.get('DROPOUT_FIXED', 0.2)),
        'hidden_dim_fixed': int(os.environ.get('HIDDEN_DIM_FIXED', 256)),
        'ehr_encoder_fixed': os.environ.get('EHR_ENCODER_FIXED', 'transformer'),
        'cxr_encoder_fixed': os.environ.get('CXR_ENCODER_FIXED', 'resnet50'),
        'ehr_n_head_fixed': int(os.environ.get('EHR_N_HEAD_FIXED', 4)),
        'ehr_n_layers_fixed': int(os.environ.get('EHR_N_LAYERS_FIXED', 1)),
        'max_len_fixed': int(os.environ.get('MAX_LEN_FIXED', 500)),
        'n_clusters_fixed': int(os.environ.get('N_CLUSTERS_FIXED', 10)),
        'seeds': [int(x) for x in os.environ.get('SEEDS', '42,123,1234').split(',')],
        
        # Finetune parameters to optimize
        'inner_loop_choices': [int(x) for x in os.environ.get('INNER_LOOP_CHOICES', '1,2,3').split(',')],
        'mc_size_choices': [int(x) for x in os.environ.get('MC_SIZE_CHOICES', '10,20,30').split(',')],
        'lr_inner_min': float(os.environ.get('LR_INNER_MIN', 0.001)),
        'lr_inner_max': float(os.environ.get('LR_INNER_MAX', 0.05)),
        'alpha_min': float(os.environ.get('ALPHA_MIN', 0.05)),
        'alpha_max': float(os.environ.get('ALPHA_MAX', 0.2)),
        'beta_min': float(os.environ.get('BETA_MIN', 0.05)),
        'beta_max': float(os.environ.get('BETA_MAX', 0.2)),
        'temperature_min': float(os.environ.get('TEMPERATURE_MIN', 1.0)),
        'temperature_max': float(os.environ.get('TEMPERATURE_MAX', 3.0))
    }
    
    # Auto-adjust num_classes for mortality task
    if config['task'] == 'mortality':
        config['num_classes'] = 1
    
    # Create and run optimizer
    optimizer = BayesianSMILOptimizer(config)
    result = optimizer.run_optimization()
    
    print("SMIL Bayesian optimization completed successfully!")

if __name__ == "__main__":
    main()
EOF
}

# Main execution
main() {
    log "Starting SMIL Bayesian Optimization Search (Finetune Parameters Only)"
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
    export INPUT_DIM="${INPUT_DIM_VALUES[0]}"
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
    
    # Fixed parameters
    export LR_FIXED="$LR_FIXED"
    export BATCH_SIZE_FIXED="$BATCH_SIZE_FIXED"
    export DROPOUT_FIXED="$DROPOUT_FIXED"
    export HIDDEN_DIM_FIXED="$HIDDEN_DIM_FIXED"
    export EHR_ENCODER_FIXED="$EHR_ENCODER_FIXED"
    export CXR_ENCODER_FIXED="$CXR_ENCODER_FIXED"
    export EHR_N_HEAD_FIXED="$EHR_N_HEAD_FIXED"
    export EHR_N_LAYERS_FIXED="$EHR_N_LAYERS_FIXED"
    export MAX_LEN_FIXED="$MAX_LEN_FIXED"
    export N_CLUSTERS_FIXED="$N_CLUSTERS_FIXED"
    export SEEDS=$(IFS=,; echo "${SEEDS[*]}")
    
    # Finetune parameters to optimize
    export INNER_LOOP_CHOICES="$INNER_LOOP_CHOICES"
    export MC_SIZE_CHOICES="$MC_SIZE_CHOICES"
    export LR_INNER_MIN="$LR_INNER_MIN"
    export LR_INNER_MAX="$LR_INNER_MAX"
    export ALPHA_MIN="$ALPHA_MIN"
    export ALPHA_MAX="$ALPHA_MAX"
    export BETA_MIN="$BETA_MIN"
    export BETA_MAX="$BETA_MAX"
    export TEMPERATURE_MIN="$TEMPERATURE_MIN"
    export TEMPERATURE_MAX="$TEMPERATURE_MAX"
    
    log "Starting Python Bayesian optimizer..."
    
    # Run the Bayesian optimizer
    cd "$BASE_DIR"
    python3 "${RESULTS_DIR}/bayesian_optimizer.py"
    
    if [ $? -eq 0 ]; then
        log "SMIL Bayesian optimization completed successfully!"
        log "Results saved to: $RESULTS_DIR"
        log "Best parameters in: $RESULTS_DIR/best_params.txt"
        log "Full results in: $RESULTS_DIR/results_summary.csv"
        log "Optimization object saved in: $RESULTS_DIR/bayesian_optimization_result.pkl"
    else
        log "SMIL Bayesian optimization failed!"
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    log "SMIL Bayesian search interrupted by user"
    exit 1
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"