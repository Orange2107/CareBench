#!/bin/bash

# ============================================================================
# FLEXMOE BAYESIAN OPTIMIZATION SEARCH - MOE ARCHITECTURE PARAMETERS
# ============================================================================

# Fold selection for bayesian search (can modify to include more folds)
SEARCH_FOLDS=(1)  

# Model Configuration
MODEL="flexmoe"
TASK="los"  # phenotype, mortality, los
GPU=1

# Basic Experiment Settings
PRETRAINED=true
USE_DEMOGRAPHICS=false
CROSS_EVAL=""  # Set to "matched_to_full" or "full_to_matched" if needed
MATCHED=true

# Bayesian Optimization Settings
N_CALLS=20                    # Total number of optimization iterations (increased for more parameters)
N_INITIAL_POINTS=5          # Number of random initial points (increased)
ACQUISITION_FUNC="gp_hedge"  # Acquisition function: 'LCB', 'EI', 'PI', 'gp_hedge'
N_JOBS=8                     # Number of parallel jobs (-1 for all cores)

# Resume settings
RESUME_FROM_CHECKPOINT=false  # Set to true to resume from previous run
CHECKPOINT_FILE=""           # Path to previous bayesian_optimization_result.pkl (auto-detect if empty)

# Search Space Bounds - FlexMoE MoE parameters
# Format: [min_value, max_value] for continuous parameters

# Core training parameters - FIXED
LR_FIXED=0.0001              # Fixed learning rate (from config)
BATCH_SIZE_FIXED=16          # Fixed batch size (from config)
EPOCHS_VALUES=(50)           # Fixed epochs (from config)
PATIENCE_VALUES=(10)         # Fixed patience (from config)

# Seeds for multiple runs
SEEDS=(42 123 1234)

# Task-specific parameters - FIXED
INPUT_DIM_VALUES=(49)                 # EHR input dimension (from config) - Note: 49 not 498
NUM_CLASSES_VALUES=(7)               # For phenotype task: 25, for mortality: 1, for los: 7 (auto-adjusted)

# FlexMoE-specific encoder parameters - FIXED
EHR_ENCODER_FIXED="transformer"       # Fixed EHR encoder (from config)
CXR_ENCODER_FIXED="resnet50"         # Fixed CXR encoder (from config)

# Architecture parameters - FIXED
HIDDEN_DIM_FIXED=256                  # Fixed hidden dimension (from config)
NUM_PATCHES_FIXED=16                  # Fixed number of patches (from config)
NUM_LAYERS_FIXED=1                    # Fixed number of layers (from config)
NUM_LAYERS_PRED_FIXED=1               # Fixed number of prediction layers (from config)
NUM_HEADS_FIXED=4                     # Fixed number of heads (from config)
DROPOUT_FIXED=0.2                     # Fixed dropout (from config)

# EHR Transformer-specific parameters - FIXED (from config)
EHR_N_HEAD_FIXED=4                    # Fixed number of attention heads
EHR_N_LAYERS_FIXED=1                  # Fixed number of layers

# EHR LSTM-specific parameters - FIXED (from config)
EHR_LSTM_BIDIRECTIONAL_FIXED=true     # Fixed bidirectional setting
EHR_LSTM_NUM_LAYERS_FIXED=1           # Fixed number of LSTM layers

# SEARCH PARAMETERS: FlexMoE MoE architecture parameters
# Based on config comments: num_experts: 2、4、8、16、32; num_routers: 1、2; gate_loss_weight: 0.001、0.01、0.1; top_k: 2、4、8、16
NUM_EXPERTS_CHOICES="4,8,16"              # Number of experts choices
NUM_ROUTERS_CHOICES="1,2"                      # Number of routers choices
TOP_K_CHOICES="2,4,8"                       # Top-k expert selection choices
GATE_LOSS_WEIGHT_MIN=0.001                     # Minimum gate loss weight
GATE_LOSS_WEIGHT_MAX=0.1                       # Maximum gate loss weight

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
    local results_dirname="${model}_${task}-${demographic_str}-${cross_eval_str}-${matched_str}-${pretrained_str}_bayesian_search_moe_params"
    
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

class BayesianFlexMoEOptimizer:
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
        
        # Define search space for FlexMoE - MoE architecture parameters
        self.dimensions = [
            Categorical(config['num_experts_choices'], name='num_experts'),
            Categorical(config['num_routers_choices'], name='num_routers'),
            Categorical(config['top_k_choices'], name='top_k'),
            Real(config['gate_loss_weight_min'], config['gate_loss_weight_max'], name='gate_loss_weight')
        ]
        
        self.dimension_names = [dim.name for dim in self.dimensions]
        
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def is_valid_flexmoe_config(self, params_dict):
        """Check if FlexMoE configuration is valid"""
        # Check if top_k is not greater than num_experts
        return params_dict['top_k'] <= params_dict['num_experts']
    
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
        # Check if configuration is valid for FlexMoE
        if not self.is_valid_flexmoe_config(params_dict):
            self.log(f"Skipping invalid config: top_k={params_dict['top_k']}, num_experts={params_dict['num_experts']} (top_k > num_experts)")
            return -1.0, -1.0  # Return poor scores for invalid configurations
        
        self.iteration += 1
        
        # Create experiment name
        exp_name = f"bayes_iter{self.iteration}_fold{fold}_lr{self.config['lr_fixed']:.6f}_bs{self.config['batch_size_fixed']}_" \
                   f"experts{params_dict['num_experts']}_routers{params_dict['num_routers']}_" \
                   f"topk{params_dict['top_k']}_gate{params_dict['gate_loss_weight']:.4f}"
        
        self.log(f"Starting Bayesian iteration {self.iteration}: {exp_name}")
        
        # Run experiments for all seeds
        all_metrics = []
        
        for seed in self.config['seeds']:
            seed_exp_name = f"{exp_name}_seed{seed}"
            self.log(f"  Running seed {seed}...")
            
            # Build command for FlexMoE with fixed parameters
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
                "--input_dim", str(self.config['input_dim']),
                "--num_classes", str(self.config['num_classes']),
                "--ehr_encoder", self.config['ehr_encoder_fixed'],
                "--cxr_encoder", self.config['cxr_encoder_fixed'],
                "--hidden_dim", str(self.config['hidden_dim_fixed']),
                "--num_patches", str(self.config['num_patches_fixed']),
                "--num_layers", str(self.config['num_layers_fixed']),
                "--num_layers_pred", str(self.config['num_layers_pred_fixed']),
                "--num_heads", str(self.config['num_heads_fixed']),
                "--dropout", str(self.config['dropout_fixed']),
                "--ehr_n_head", str(self.config['ehr_n_head_fixed']),
                "--ehr_n_layers", str(self.config['ehr_n_layers_fixed']),
                "--ehr_num_layers", str(self.config['ehr_lstm_num_layers_fixed']),
                "--num_experts", str(params_dict['num_experts']),     # The parameter we're optimizing
                "--num_routers", str(params_dict['num_routers']),    # The parameter we're optimizing
                "--top_k", str(params_dict['top_k']),                # The parameter we're optimizing
                "--gate_loss_weight", str(params_dict['gate_loss_weight']),  # The parameter we're optimizing
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

            # Handle boolean parameters
            if self.config['ehr_lstm_bidirectional_fixed']:
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
                'lr_fixed': self.config['lr_fixed'],
                'batch_size_fixed': self.config['batch_size_fixed'],
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
        
        # Validate FlexMoE configuration first
        if not self.is_valid_flexmoe_config(params_dict):
            # Return very poor score for invalid configurations
            return 1.0  # High value because we minimize
        
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
        self.log("Starting Bayesian Optimization for FlexMoE (MoE architecture parameters)")
        self.log(f"Fixed Learning Rate: {self.config['lr_fixed']}")
        self.log(f"Fixed Batch Size: {self.config['batch_size_fixed']}")
        self.log(f"Seeds: {self.config['seeds']}")
        self.log(f"Search space: num_experts={self.config['num_experts_choices']}, num_routers={self.config['num_routers_choices']}")
        self.log(f"              top_k={self.config['top_k_choices']}, gate_loss_weight=[{self.config['gate_loss_weight_min']}, {self.config['gate_loss_weight_max']}]")
        
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
            f.write("FlexMoE Bayesian Optimization Best Parameters (MoE Architecture)\n")
            f.write("=" * 65 + "\n")
            f.write(f"Best ACC: {self.best_score:.4f}\n")
            f.write(f"Total iterations: {self.iteration}\n\n")
            f.write("Best Parameters:\n")
            if self.best_params:
                for param, value in self.best_params.items():
                    f.write(f"  {param}: {value}\n")
            f.write(f"Fixed Learning Rate: {self.config['lr_fixed']}\n")
            f.write(f"Fixed Batch Size: {self.config['batch_size_fixed']}\n")
            f.write(f"Seeds used: {self.config['seeds']}\n")
        
        # Generate convergence plot
        self.generate_convergence_plot(result)
        
        return result
    
    def generate_convergence_plot(self, result):
        """Generate convergence plot"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(18, 12), facecolor='white')
            
            # Plot convergence
            scores = [-y for y in result.func_vals]  # Convert back to positive
            best_scores = [max(scores[:i+1]) for i in range(len(scores))]
            
            plt.subplot(3, 3, 1)
            plt.plot(scores, 'bo-', alpha=0.6, label='ACC')
            plt.plot(best_scores, 'r-', linewidth=2, label='Best ACC')
            plt.xlabel('Iteration')
            plt.ylabel('ACC')
            plt.title('FlexMoE Bayesian Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot parameter exploration
            if hasattr(result, 'x_iters') and len(result.x_iters) > 1:
                param_data = pd.DataFrame(result.x_iters, columns=self.dimension_names)
                
                # num_experts exploration
                plt.subplot(3, 3, 2)
                plt.plot(param_data['num_experts'], 'g-o', alpha=0.7)
                plt.xlabel('Iteration')
                plt.ylabel('Number of Experts')
                plt.title('Number of Experts Exploration')
                plt.grid(True, alpha=0.3)
                
                # num_routers exploration
                plt.subplot(3, 3, 3)
                plt.plot(param_data['num_routers'], 'b-o', alpha=0.7)
                plt.xlabel('Iteration')
                plt.ylabel('Number of Routers')
                plt.title('Number of Routers Exploration')
                plt.grid(True, alpha=0.3)
                
                # top_k exploration
                plt.subplot(3, 3, 4)
                plt.plot(param_data['top_k'], 'purple', marker='o', alpha=0.7)
                plt.xlabel('Iteration')
                plt.ylabel('Top-K')
                plt.title('Top-K Expert Selection Exploration')
                plt.grid(True, alpha=0.3)
                
                # gate_loss_weight exploration
                plt.subplot(3, 3, 5)
                plt.plot(param_data['gate_loss_weight'], 'orange', marker='o', alpha=0.7)
                plt.xlabel('Iteration')
                plt.ylabel('Gate Loss Weight')
                plt.title('Gate Loss Weight Exploration')
                plt.grid(True, alpha=0.3)
                
                # num_experts vs top_k relationship
                plt.subplot(3, 3, 6)
                for i, (experts, topk) in enumerate(zip(param_data['num_experts'], param_data['top_k'])):
                    color = 'green' if topk <= experts else 'red'
                    plt.scatter(experts, topk, c=color, alpha=0.6)
                plt.xlabel('Number of Experts')
                plt.ylabel('Top-K')
                plt.title('Experts vs Top-K (Green=Valid, Red=Invalid)')
                plt.grid(True, alpha=0.3)
                
                # Performance vs num_experts
                plt.subplot(3, 3, 7)
                plt.scatter(param_data['num_experts'], scores, alpha=0.7, c=range(len(scores)), cmap='viridis')
                plt.colorbar(label='Iteration')
                plt.xlabel('Number of Experts')
                plt.ylabel('ACC')
                plt.title('Performance vs Number of Experts')
                plt.grid(True, alpha=0.3)
                
                # Performance vs gate_loss_weight
                plt.subplot(3, 3, 8)
                plt.scatter(param_data['gate_loss_weight'], scores, alpha=0.7, c=range(len(scores)), cmap='plasma')
                plt.colorbar(label='Iteration')
                plt.xlabel('Gate Loss Weight')
                plt.ylabel('ACC')
                plt.title('Performance vs Gate Loss Weight')
                plt.grid(True, alpha=0.3)
                
                # 3D scatter plot: experts vs top_k vs performance
                from mpl_toolkits.mplot3d import Axes3D
                ax = plt.subplot(3, 3, 9, projection='3d')
                scatter = ax.scatter(param_data['num_experts'], param_data['top_k'], 
                                   param_data['gate_loss_weight'], c=scores, cmap='viridis', alpha=0.7)
                ax.set_xlabel('Number of Experts')
                ax.set_ylabel('Top-K')
                ax.set_zlabel('Gate Loss Weight')
                ax.set_title('3D Parameter Space')
                plt.colorbar(scatter, ax=ax, label='ACC', shrink=0.5)
            
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
        'n_calls': int(os.environ.get('N_CALLS', 25)),
        'n_initial_points': int(os.environ.get('N_INITIAL_POINTS', 8)),
        'acquisition_func': os.environ.get('ACQUISITION_FUNC', 'gp_hedge'),
        'n_jobs': int(os.environ.get('N_JOBS', 1)),
        
        # Resume parameters
        'resume_from_checkpoint': os.environ.get('RESUME_FROM_CHECKPOINT', 'false').lower() == 'true',
        'checkpoint_file': os.environ.get('CHECKPOINT_FILE', ''),
        
        # Fixed parameters from FlexMoE config
        'lr_fixed': float(os.environ.get('LR_FIXED', 0.0001)),
        'batch_size_fixed': int(os.environ.get('BATCH_SIZE_FIXED', 16)),
        'seeds': [int(x) for x in os.environ.get('SEEDS', '42,123,1234').split(',')],
        
        # FlexMoE-specific fixed parameters
        'ehr_encoder_fixed': os.environ.get('EHR_ENCODER_FIXED', 'transformer'),
        'cxr_encoder_fixed': os.environ.get('CXR_ENCODER_FIXED', 'resnet50'),
        'hidden_dim_fixed': int(os.environ.get('HIDDEN_DIM_FIXED', 256)),
        'num_patches_fixed': int(os.environ.get('NUM_PATCHES_FIXED', 16)),
        'num_layers_fixed': int(os.environ.get('NUM_LAYERS_FIXED', 1)),
        'num_layers_pred_fixed': int(os.environ.get('NUM_LAYERS_PRED_FIXED', 1)),
        'num_heads_fixed': int(os.environ.get('NUM_HEADS_FIXED', 4)),
        'dropout_fixed': float(os.environ.get('DROPOUT_FIXED', 0.2)),
        'ehr_n_head_fixed': int(os.environ.get('EHR_N_HEAD_FIXED', 4)),
        'ehr_n_layers_fixed': int(os.environ.get('EHR_N_LAYERS_FIXED', 1)),
        'ehr_lstm_num_layers_fixed': int(os.environ.get('EHR_LSTM_NUM_LAYERS_FIXED', 1)),
        'ehr_lstm_bidirectional_fixed': os.environ.get('EHR_LSTM_BIDIRECTIONAL_FIXED', 'true').lower() == 'true',
        
        # Search parameter choices and bounds
        'num_experts_choices': [int(x) for x in os.environ.get('NUM_EXPERTS_CHOICES', '2,4,8,16,32').split(',')],
        'num_routers_choices': [int(x) for x in os.environ.get('NUM_ROUTERS_CHOICES', '1,2').split(',')],
        'top_k_choices': [int(x) for x in os.environ.get('TOP_K_CHOICES', '2,4,8,16').split(',')],
        'gate_loss_weight_min': float(os.environ.get('GATE_LOSS_WEIGHT_MIN', 0.001)),
        'gate_loss_weight_max': float(os.environ.get('GATE_LOSS_WEIGHT_MAX', 0.1))
    }
    
    # Auto-adjust num_classes for mortality task
    if config['task'] == 'mortality':
        config['num_classes'] = 1
    
    # Create and run optimizer
    optimizer = BayesianFlexMoEOptimizer(config)
    result = optimizer.run_optimization()
    
    print("FlexMoE Bayesian optimization completed successfully!")

if __name__ == "__main__":
    main()
EOF
}

# Main execution
main() {
    log "Starting FlexMoE Bayesian Optimization Search (MoE architecture parameters)"
    log "Configuration: MODEL=$MODEL, TASK=$TASK, USE_DEMOGRAPHICS=$USE_DEMOGRAPHICS, CROSS_EVAL=$CROSS_EVAL, PRETRAINED=$PRETRAINED"
    log "Results will be saved to: $RESULTS_DIR"
    log "Log file: $LOG_FILE"
    log "Total optimization calls: $N_CALLS"
    log "Initial random points: $N_INITIAL_POINTS"
    log "Acquisition function: $ACQUISITION_FUNC"
    log "Search parameters: num_experts=$NUM_EXPERTS_CHOICES, num_routers=$NUM_ROUTERS_CHOICES"
    log "                   top_k=$TOP_K_CHOICES, gate_loss_weight=[$GATE_LOSS_WEIGHT_MIN, $GATE_LOSS_WEIGHT_MAX]"
    
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
    export SEEDS=$(IFS=,; echo "${SEEDS[*]}")
    
    # FlexMoE-specific fixed parameters
    export EHR_ENCODER_FIXED="$EHR_ENCODER_FIXED"
    export CXR_ENCODER_FIXED="$CXR_ENCODER_FIXED"
    export HIDDEN_DIM_FIXED="$HIDDEN_DIM_FIXED"
    export NUM_PATCHES_FIXED="$NUM_PATCHES_FIXED"
    export NUM_LAYERS_FIXED="$NUM_LAYERS_FIXED"
    export NUM_LAYERS_PRED_FIXED="$NUM_LAYERS_PRED_FIXED"
    export NUM_HEADS_FIXED="$NUM_HEADS_FIXED"
    export DROPOUT_FIXED="$DROPOUT_FIXED"
    export EHR_N_HEAD_FIXED="$EHR_N_HEAD_FIXED"
    export EHR_N_LAYERS_FIXED="$EHR_N_LAYERS_FIXED"
    export EHR_LSTM_NUM_LAYERS_FIXED="$EHR_LSTM_NUM_LAYERS_FIXED"
    export EHR_LSTM_BIDIRECTIONAL_FIXED="$EHR_LSTM_BIDIRECTIONAL_FIXED"
    
    # Search parameter bounds
    export NUM_EXPERTS_CHOICES="$NUM_EXPERTS_CHOICES"
    export NUM_ROUTERS_CHOICES="$NUM_ROUTERS_CHOICES"
    export TOP_K_CHOICES="$TOP_K_CHOICES"
    export GATE_LOSS_WEIGHT_MIN="$GATE_LOSS_WEIGHT_MIN"
    export GATE_LOSS_WEIGHT_MAX="$GATE_LOSS_WEIGHT_MAX"
    
    log "Starting Python Bayesian optimizer..."
    
    # Run the Bayesian optimizer
    cd "$BASE_DIR"
    python3 "${RESULTS_DIR}/bayesian_optimizer.py"
    
    if [ $? -eq 0 ]; then
        log "FlexMoE Bayesian optimization completed successfully!"
        log "Results saved to: $RESULTS_DIR"
        log "Best parameters in: $RESULTS_DIR/best_params.txt"
        log "Full results in: $RESULTS_DIR/results_summary.csv"
        log "Optimization object saved in: $RESULTS_DIR/bayesian_optimization_result.pkl"
    else
        log "FlexMoE Bayesian optimization failed!"
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    log "FlexMoE Bayesian search interrupted by user"
    exit 1
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"