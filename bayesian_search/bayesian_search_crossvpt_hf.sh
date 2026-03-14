#!/bin/bash

# ============================================================================
# CROSSVPT BAYESIAN OPTIMIZATION SEARCH - MODIFY THESE VALUES TO CUSTOMIZE THE SEARCH
# ============================================================================

# ✨ NEW: CLEANUP-ONLY MODE
# Set CLEANUP_ONLY=true to skip experiments and only cleanup checkpoints
# This is useful when you want to:
#   1. Delete checkpoints from non-best hyperparameter combinations
#   2. Keep only the best iteration's checkpoints
#   3. Re-run after interrupted experiments without re-running completed ones
#
# Usage:
#   Normal mode:     CLEANUP_ONLY=false  (default)
#   Cleanup mode:    CLEANUP_ONLY=true
#
# Example:
#   # Run cleanup only (no new experiments)
#   CLEANUP_ONLY=true ./bayesian_search_crossvpt_hf.sh
#
#   # Or edit this file and set: CLEANUP_ONLY=true

'''
python main.py \
  --model crossvpt \
  --mode train \
  --task phenotype \
  --fold 1 \
  --gpu 0 \
  --lr 0.0001 \
  --batch_size 16 \
  --epochs 50 \
  --patience 10 \
  --cxr_encoder hf_chexpert_vit \
  --hidden_size 256 \
  --ehr_dropout 0.2 \
  --ehr_n_head 4 \
  --ehr_n_layers_distinct 1 \
  --num_prompt_tokens 10 \
  --prompt_token_dropout 0.055 \
  --prompt_noise_std 0.078 \
  --loss_multi 1.0 \
  --loss_ehr 1.0 \
  --loss_cxr 1.0 \
  --aux_ehr_weight 0.0 \
  --seed 42 \
  --log_dir /hdd/bayesian_search_experiments/crossvpt/phenotype \
  --vpt_feature hf_chexpert_vit \
  --hf_model_id codewithdark/vit-chest-xray \
  --freeze_vit false \
  --bias_tune false \
  --partial_layers 0 \
  --fusion_variant null \
  --final_pred_mode concat
'''




# Fold selection for bayesian search (can modify to include more folds)
SEARCH_FOLDS=(1)  

# Model Configuration
MODEL="crossvpt"
TASK="phenotype"  # phenotype, mortality, or los
GPU="0,1,2"  # 支持多 GPU 并行，用逗号分隔（如 "0,1,2" 表示 3 个 GPU）

# Basic Experiment Settings
PRETRAINED=true
USE_DEMOGRAPHICS=false
CROSS_EVAL=""  # Set to "matched_to_full" or "full_to_matched" if needed
MATCHED=false
INPUT_DIM=498  # EHR input dimension (fixed)

# Bayesian Optimization Settings
N_CALLS=20                    # Total number of optimization iterations
N_INITIAL_POINTS=5            # Number of random initial points
ACQUISITION_FUNC="gp_hedge"   # Acquisition function: 'LCB', 'EI', 'PI', 'gp_hedge'
N_JOBS=8                      # Number of parallel jobs (-1 for all cores)

# Resume settings
RESUME_FROM_CHECKPOINT=false  # Set to true to resume from previous run
CHECKPOINT_FILE=""            # Path to previous bayesian_optimization_result.pkl (auto-detect if empty)

# ✨ NEW: Cleanup-only mode (skip experiments, only cleanup checkpoints)
CLEANUP_ONLY=true          # Set to true to only cleanup checkpoints without running new experiments

# Search Space Bounds - Define parameter ranges for Bayesian optimization
# FIXED PARAMETERS (same as other baselines):
# - batch_size: 16
# - learning_rate: 0.0001
# - ehr_encoder: transformer
# - cxr_encoder: hf_chexpert_vit
# - hidden_size: 256
# - ehr_dropout: 0.2
# - ehr_n_head: 4
# - ehr_n_layers_distinct: 1

# Core training parameters - FIXED
LR_FIXED=0.0001              # Fixed learning rate (same as baselines)
BATCH_SIZE_CHOICES="16"      # Fixed batch size (same as baselines)
EPOCHS_VALUES=(50)           # Keep epochs reasonable
PATIENCE_VALUES=(10)         # Patience for early stopping

# Seeds for multiple runs
SEEDS=(42 123 1234)

# Task-specific parameters
NUM_CLASSES_VALUES=(9)       # For phenotype: 25, mortality: 1, los: 7 (auto-adjusted)

# CrossVPT-specific encoder parameters (FIXED to match baselines)
EHR_ENCODER_CHOICES="transformer"    # EHR encoder: transformer (fixed)
CXR_ENCODER_CHOICES="hf_chexpert_vit"  # HF ViT encoder (fixed)
FREEZE_VIT=false                      # keep ViT training strategy fixed across search
BIAS_TUNE=false
PARTIAL_LAYERS=0

# Architecture parameters (FIXED to match baselines)
HIDDEN_SIZE_CHOICES="256"         # Hidden dimension (fixed to match baselines)
EHR_DROPOUT_FIXED=0.2             # Fixed dropout rate (same as baselines)

# EHR Transformer-specific parameters (FIXED to match baselines)
EHR_N_HEAD_CHOICES="4"                 # Number of attention heads (fixed)
EHR_N_LAYERS_DISTINCT_CHOICES="1"     # Distinct layers (fixed)

# Loss weights - FIXED (与 baseline 保持一致)
LOSS_MULTI_FIXED=1.0          # Multi-modal loss weight
LOSS_EHR_FIXED=1.0            # EHR auxiliary loss weight  
LOSS_CXR_FIXED=1.0            # CXR auxiliary loss weight
AUX_EHR_WEIGHT_FIXED=0.0      # Keep aligned with configs/crossvpt.yaml


# CrossVPT-specific: Prompt parameters (ONLY THESE ARE SEARCHED!)
NUM_PROMPT_TOKENS_CHOICES="3,5,7,10"     # Number of prompt tokens (key parameter!)
PROMPT_TOKEN_DROPOUT_MIN=0.0              # Minimum prompt token dropout
PROMPT_TOKEN_DROPOUT_MAX=0.3              # Maximum prompt token dropout
PROMPT_NOISE_STD_MIN=0.0                  # Minimum prompt noise std
PROMPT_NOISE_STD_MAX=0.1                  # Maximum prompt noise std


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
    
    echo "/hdd/bayesian_search_experiments/${model}/${task}/lightning_logs/${results_dirname}"
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
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class BayesianCrossVPTOptimizer:
    def __init__(self, config):
        self.config = config
        self.results_dir = config['results_dir']
        self.log_file = config['log_file']
        self.iteration = 0
        self.best_score = -np.inf
        self.best_params = None
        self.best_iteration = None
        
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
                            if 'prauc_mean' in prev_df.columns:
                                best_row = prev_df.loc[prev_df['prauc_mean'].idxmax()]
                                self.best_score = best_row['prauc_mean']
                            
                    self.log(f"Resuming from iteration {self.iteration}, best PRAUC so far: {self.best_score:.4f}")
                    
                except Exception as e:
                    self.log(f"Failed to load checkpoint: {e}, starting fresh")
                    self.previous_result = None
            else:
                self.log(f"Checkpoint file not found: {checkpoint_file}, starting fresh")
        
        # ✨ NEW: Scan for existing experiments and skip already-run parameter combinations
        self.already_run_params = set()
        self.scan_existing_experiments()
        
        # Define search space for CrossVPT - ONLY PROMPT PARAMETERS
        # Loss weights are FIXED to match baselines (LOSS_MULTI_FIXED, LOSS_EHR_FIXED, etc.)

        # 定义搜索空间
        self.dimensions = [
            Categorical(config['num_prompt_tokens_choices'], name='num_prompt_tokens'),
            Real(config['prompt_token_dropout_min'], config['prompt_token_dropout_max'], name='prompt_token_dropout'),
            Real(config['prompt_noise_std_min'], config['prompt_noise_std_max'], name='prompt_noise_std'),
        ]
        
        self.dimension_names = [dim.name for dim in self.dimensions]
    
    def scan_existing_experiments(self):
        """
        ✨ NEW: Scan for existing experiments and extract their parameters.
        This allows skipping already-run parameter combinations.
        """
        if not os.path.exists(self.results_dir):
            return
        
        self.log("🔍 Scanning for existing experiments...")
        
        # Look for results_summary.csv
        csv_file = os.path.join(self.results_dir, "results_summary.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    # Extract parameter tuple: (num_prompt_tokens, prompt_token_dropout, prompt_noise_std)
                    params_key = (
                        int(row['num_prompt_tokens']),
                        float(row['prompt_token_dropout']),
                        float(row['prompt_noise_std'])
                    )
                    self.already_run_params.add(params_key)
                
                self.log(f"✅ Found {len(self.already_run_params)} existing experiment(s) in results_summary.csv")
                
                # Find best existing result
                if 'PRAUC_mean' in df.columns and df['PRAUC_mean'].notna().any():
                    best_row = df.loc[df['PRAUC_mean'].idxmax()]
                    self.best_score = best_row['PRAUC_mean']
                    self.best_iteration = int(best_row['iteration'])
                    
                    # Extract best params
                    self.best_params = {
                        'num_prompt_tokens': int(best_row['num_prompt_tokens']),
                        'prompt_token_dropout': float(best_row['prompt_token_dropout']),
                        'prompt_noise_std': float(best_row['prompt_noise_std']),
                        'batch_size': int(best_row.get('batch_size', 16))
                    }
                    self.log(f"✨ Best existing result: Iteration {self.best_iteration}, PRAUC={self.best_score:.4f}")
                    self.log(f"   Best params: {self.best_params}")
                    
            except Exception as e:
                self.log(f"⚠️  Failed to read results_summary.csv: {e}")
        else:
            # Fallback: scan directory names for experiment patterns
            self.log("📁 results_summary.csv not found, scanning directory names...")
            import re
            
            # Pattern: CROSSVPT-model_crossvpt-task_phenotype-fold_1-...-num_prompt_tokens_{N}-prompt_token_dropout_{D}-prompt_noise_std_{N}-...
            pattern = r'num_prompt_tokens_(\d+)-prompt_token_dropout_([0-9.]+)-prompt_noise_std_([0-9.]+)'
            
            lightning_logs_root = self.get_lightning_logs_root()
            if os.path.exists(lightning_logs_root):
                for dir_name in os.listdir(lightning_logs_root):
                    match = re.search(pattern, dir_name)
                    if match:
                        params_key = (
                            int(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3))
                        )
                        self.already_run_params.add(params_key)
                
                self.log(f"✅ Found {len(self.already_run_params)} existing experiment(s) from directory scan")
    
    def cleanup_existing_checkpoints_only(self):
        """
        ✨ NEW: Only cleanup checkpoints from existing experiments without running new ones.
        Keeps only the best iteration's checkpoints.
        """
        self.log("\n" + "="*60)
        self.log("🧹 CHECKPOINT CLEANUP MODE (No new experiments)")
        self.log("="*60)
        
        if not self.results_data:
            self.log("⚠️  No results data to cleanup")
            return
        
        if self.best_iteration is None:
            self.log("⚠️  No best iteration found, cannot cleanup")
            return
        
        self.log(f"✅ Will keep checkpoints for best iteration: {self.best_iteration}")
        self.log(f"   Best PRAUC: {self.best_score:.4f}")
        self.log(f"   Best params: {self.best_params}")
        
        deleted_count = 0
        kept_count = 0
        
        for result in self.results_data:
            iteration = result['iteration']
            if iteration == self.best_iteration:
                kept_count += 1
                self.log(f"   ⏭️  Keeping iteration {iteration}")
            else:
                deleted = self.delete_iteration_checkpoints(
                    result['experiment_name'],
                    iteration_label=iteration,
                )
                if deleted > 0:
                    deleted_count += 1
        
        self.log(f"\n💾 CLEANUP SUMMARY")
        self.log(f"   Deleted {deleted_count} non-best iteration(s)")
        self.log(f"   Kept {kept_count} best iteration(s)")
        self.log("="*60 + "\n")
        
        # ✨ NEW: Save detailed metrics for best iteration's three seeds
        self.save_best_iteration_detailed_metrics()
    
    def save_best_iteration_detailed_metrics(self):
        """
        ✨ NEW: Save detailed metrics from all three seeds of the best iteration.
        Creates a CSV with mean ± std format for all metrics in test_set_results.yaml
        """
        if self.best_iteration is None:
            self.log("⚠️  No best iteration found, skipping detailed metrics save")
            return
        
        self.log("\n" + "="*60)
        self.log("📊 SAVING DETAILED METRICS FOR BEST ITERATION")
        self.log("="*60)
        self.log(f"Best iteration: {self.best_iteration}")
        
        # ✨ FIXED: Get the actual params from results_data for the best iteration
        best_result = None
        for result in self.results_data:
            if result['iteration'] == self.best_iteration:
                best_result = result
                break
        
        if not best_result:
            self.log("⚠️  Could not find result data for best iteration")
            return
        
        # Extract actual params from the result
        best_num_tokens = int(best_result['num_prompt_tokens'])
        best_dropout = float(best_result['prompt_token_dropout'])
        best_noise = float(best_result['prompt_noise_std'])
        
        self.log(f"Best params: num_tokens={best_num_tokens}, dropout={best_dropout}, noise={best_noise}")
        
        # Collect metrics from all seeds of the best iteration.
        # Use the exact experiment_name + seed logs to resolve real run dirs,
        # instead of fuzzy matching directory names (which can be truncated/hashed).
        best_exp_name = best_result.get('experiment_name')
        if not best_exp_name:
            self.log("⚠️  best_result missing experiment_name, cannot resolve seed runs")
            return

        self.log(f"Resolving seed runs from experiment: {best_exp_name}")

        all_seed_metrics = {}
        seeds_found = []
        
        for seed in self.config['seeds']:
            found_dir = self.get_real_training_dir(best_exp_name, seed)
            if not found_dir:
                self.log(f"⚠️  Could not resolve real training directory for seed {seed}")
                continue

            self.log(f"✅ Using seed {seed} dir: {found_dir}")
            
            # Look for test_set_results.yaml
            test_results_file = os.path.join(found_dir, "test_set_results.yaml")
            if not os.path.exists(test_results_file):
                self.log(f"⚠️  test_set_results.yaml not found: {test_results_file}")
                continue
            
            # Load YAML file
            try:
                import yaml
                with open(test_results_file, 'r') as f:
                    metrics = yaml.safe_load(f)
                
                if metrics:
                    all_seed_metrics[seed] = metrics
                    seeds_found.append(seed)
                    self.log(f"   Loaded {len(metrics)} metrics")
            except Exception as e:
                self.log(f"⚠️  Failed to load test_set_results.yaml for seed {seed}: {e}")
        
        if not all_seed_metrics:
            self.log("⚠️  No metrics loaded from any seed")
            return
        
        self.log(f"\n📈 Successfully loaded metrics from {len(seeds_found)} seeds: {seeds_found}")
        
        # Collect all unique metric keys
        all_metric_keys = set()
        for seed_metrics in all_seed_metrics.values():
            all_metric_keys.update(seed_metrics.keys())
        
        self.log(f"📊 Found {len(all_metric_keys)} unique metric keys")
        
        # Calculate mean and std for each metric
        detailed_results = []
        for metric_key in sorted(all_metric_keys):
            values = []
            for seed_metrics in all_seed_metrics.values():
                if metric_key in seed_metrics:
                    values.append(seed_metrics[metric_key])
            
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values) if len(values) > 1 else 0.0
                
                # Format as "mean ± std"
                formatted = f"{mean_val:.4f} ± {std_val:.4f}"
                
                detailed_results.append({
                    'metric': metric_key,
                    'mean': mean_val,
                    'std': std_val,
                    'formatted': formatted,
                    'seeds_count': len(values)
                })
        
        # Save to CSV
        if detailed_results:
            csv_file = os.path.join(self.results_dir, "best_iteration_detailed_metrics.csv")
            df = pd.DataFrame(detailed_results)
            df.to_csv(csv_file, index=False)
            
            self.log(f"\n✅ Saved detailed metrics to: {csv_file}")
            self.log(f"   Total metrics: {len(detailed_results)}")
            
            # Show preview of overall metrics
            self.log(f"\n📊 Preview (overall metrics):")
            for row in detailed_results:
                if row['metric'].startswith('overall/'):
                    self.log(f"   {row['metric']}: {row['formatted']}")
            
            # Verify PRAUC calculation
            if 'overall/PRAUC' in all_metric_keys:
                prauc_values = [all_seed_metrics[seed]['overall/PRAUC'] for seed in seeds_found if 'overall/PRAUC' in all_seed_metrics[seed]]
                if len(prauc_values) > 1:
                    self.log(f"\n🔍 PRAUC Verification:")
                    for i, seed in enumerate(seeds_found):
                        if 'overall/PRAUC' in all_seed_metrics[seed]:
                            self.log(f"   Seed {seed}: {all_seed_metrics[seed]['overall/PRAUC']:.6f}")
                    self.log(f"   Mean: {np.mean(prauc_values):.4f}")
                    self.log(f"   Std:  {np.std(prauc_values):.4f}")
                    self.log(f"   Result: {np.mean(prauc_values):.4f} ± {np.std(prauc_values):.4f}")
        else:
            self.log("⚠️  No metrics to save")
        
        self.log("="*60 + "\n")
        
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def get_lightning_logs_root(self):
        """Parent lightning_logs directory that stores real training runs."""
        return os.path.dirname(self.results_dir)

    def _resolve_seed_exp_dir(self, exp_name, seed):
        """Resolve seed experiment directory under results_dir with robust fallback."""
        exact_dir = os.path.join(self.results_dir, f"{exp_name}_seed{seed}")
        if os.path.isdir(exact_dir):
            return exact_dir

        # Fallback: match by iteration/fold/seed and choose nearest params.
        base_match = re.search(r'bayes_iter(\d+)_fold(\d+)', str(exp_name))
        param_match = re.search(r'prompt_(\d+)_ptd([0-9.]+)_noise([0-9.]+)', str(exp_name))
        if not base_match:
            return None

        target_iter = int(base_match.group(1))
        target_fold = int(base_match.group(2))
        target_prompt = int(param_match.group(1)) if param_match else None
        target_ptd = float(param_match.group(2)) if param_match else None
        target_noise = float(param_match.group(3)) if param_match else None

        candidates = []
        if not os.path.isdir(self.results_dir):
            return None

        for name in os.listdir(self.results_dir):
            path = os.path.join(self.results_dir, name)
            if not os.path.isdir(path):
                continue
            if not name.startswith(f"bayes_iter{target_iter}_fold{target_fold}_"):
                continue
            if not name.endswith(f"_seed{seed}"):
                continue

            m = re.search(r'prompt_(\d+)_ptd([0-9.]+)_noise([0-9.]+)', name)
            if not m:
                continue
            prompt = int(m.group(1))
            ptd = float(m.group(2))
            noise = float(m.group(3))

            dist = 0.0
            if target_prompt is not None:
                dist += abs(prompt - target_prompt) * 10.0
            if target_ptd is not None:
                dist += abs(ptd - target_ptd)
            if target_noise is not None:
                dist += abs(noise - target_noise)

            candidates.append((dist, path))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def extract_version_name_from_seed_log(self, exp_name, seed):
        """Read the real Lightning version name from one seed's output log."""
        seed_dir = self._resolve_seed_exp_dir(exp_name, seed)
        if not seed_dir:
            return None

        seed_log = os.path.join(seed_dir, "output.log")
        if not os.path.exists(seed_log):
            return None

        try:
            with open(seed_log, 'r') as f:
                content = f.read()
            match = re.search(r"in the ver_name (.+)", content)
            if match:
                return match.group(1).strip()
        except Exception as e:
            self.log(f"⚠️  Failed to parse version name from {seed_log}: {e}")

        return None

    def get_real_training_dir(self, exp_name, seed):
        """Resolve the real Lightning run directory for one seed."""
        version_name = self.extract_version_name_from_seed_log(exp_name, seed)
        if not version_name:
            return None

        candidate = os.path.join(self.get_lightning_logs_root(), version_name)
        if os.path.isdir(candidate):
            return candidate
        return None

    def build_best_export_name(self, result_row):
        """Create a stable BEST_ directory name for the selected best setting."""
        prompt_tokens = result_row.get('num_prompt_tokens')
        prompt_dropout = result_row.get('prompt_token_dropout')
        prompt_noise = result_row.get('prompt_noise_std')
        fold = result_row.get('fold')
        prauc = result_row.get('PRAUC_mean')

        return (
            f"BEST_{self.config['model']}_{self.config['task']}_"
            f"fold{fold}_"
            f"prompt{prompt_tokens}_"
            f"ptd{prompt_dropout:.3f}_"
            f"noise{prompt_noise:.3f}_"
            f"prauc{prauc:.4f}"
        )

    def export_best_experiment(self):
        """Copy the best real training directory into a stable BEST_ directory."""
        if self.best_iteration is None or not self.results_data:
            self.log("No best iteration available; skipping BEST export")
            return

        best_rows = [row for row in self.results_data if row['iteration'] == self.best_iteration]
        if not best_rows:
            self.log(f"Best iteration {self.best_iteration} not found in results data; skipping BEST export")
            return

        best_row = best_rows[-1]
        target_dir = os.path.join(self.results_dir, self.build_best_export_name(best_row))

        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        os.makedirs(target_dir, exist_ok=True)

        exported_seeds = {}
        for seed in self.config['seeds']:
            source_dir = self.get_real_training_dir(best_row['experiment_name'], seed)
            if not source_dir:
                self.log(
                    f"⚠️  Could not resolve real training directory for best iteration "
                    f"{self.best_iteration}, seed {seed}; skipping this seed"
                )
                continue

            seed_target_dir = os.path.join(target_dir, f"seed{seed}")
            shutil.copytree(source_dir, seed_target_dir)
            exported_seeds[str(seed)] = {
                'source_dir': source_dir,
                'exported_dir': seed_target_dir,
            }

        if not exported_seeds:
            self.log("Could not export any seed directories for the best iteration")
            shutil.rmtree(target_dir)
            return

        metadata = {
            'best_iteration': self.best_iteration,
            'best_experiment_name': best_row['experiment_name'],
            'best_score': self.best_score,
            'exported_seeds': exported_seeds,
            'best_params': self.best_params,
            'all_seed_scores': best_row.get('all_metrics', []),
        }
        metadata_file = os.path.join(target_dir, "best_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.log(f"✅ Exported BEST experiment (all seeds) to: {target_dir}")
    
    def delete_iteration_checkpoints(self, exp_name, iteration_label=None):
        """Delete checkpoints for all seeds of one Bayesian iteration."""
        deleted_count = 0

        for seed in self.config['seeds']:
            checkpoint_base_dir = self.get_real_training_dir(exp_name, seed)
            if not checkpoint_base_dir:
                self.log(f"   ⚠️  No matching directory found for iter {iteration_label} seed {seed}")
                continue

            checkpoint_dir = os.path.join(checkpoint_base_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                try:
                    shutil.rmtree(checkpoint_dir)
                    deleted_count += 1
                    if iteration_label is not None:
                        self.log(f"   🗑️  Deleted iter {iteration_label} seed {seed}: {checkpoint_base_dir}")
                    else:
                        self.log(f"   🗑️  Deleted checkpoints for seed {seed}: {checkpoint_base_dir}")
                except Exception as e:
                    self.log(f"   ⚠️  Failed to delete {checkpoint_dir}: {e}")

        if deleted_count > 0:
            saved_space_mb = deleted_count * 500
            self.log(f"💾 Freed ~{saved_space_mb}MB disk space")
        return deleted_count
    
    def extract_metrics_from_log(self, log_file):
        """Extract all performance metrics from experiment log"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            metrics = {}
            
            # Extract all metrics using regex patterns
            patterns = {
                'PRAUC': r"overall/PRAUC:\s*([0-9]+\.[0-9]+)",
                'ROC_AUC': r"overall/ROC_AUC:\s*([0-9]+\.[0-9]+)",
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
        """Run experiment with multiple seeds in parallel across multiple GPUs"""
        self.iteration += 1
        
        # Create experiment name
        exp_name = f"bayes_iter{self.iteration}_fold{fold}_lr{self.config['lr_fixed']:.6f}_bs{params_dict['batch_size']}_" \
                   f"prompt_{params_dict['num_prompt_tokens']}_ptd{params_dict['prompt_token_dropout']:.3f}_" \
                   f"noise{params_dict['prompt_noise_std']:.3f}"
        
        self.log(f"Starting Bayesian iteration {self.iteration}: {exp_name}")
        
        # Parse available GPUs
        gpu_list = [int(g) for g in str(self.config['gpus']).split(',')]
        num_gpus = len(gpu_list)
        self.log(f"🚀 Using {num_gpus} GPUs for parallel execution: {gpu_list}")
        
        # Track experiment directories for potential cleanup
        current_seed_dirs = []
        
        # Initialize metrics collection
        all_metrics = []
        
        # ✨ MULTI-GPU PARALLEL EXECUTION using subprocess (no multiprocessing)
        
        # Prepare tasks for all seeds
        seed_tasks = []
        for i, seed in enumerate(self.config['seeds']):
            gpu_id = gpu_list[i % num_gpus]  # Round-robin GPU assignment
            seed_exp_name = f"{exp_name}_seed{seed}"
            seed_exp_dir = os.path.join(self.results_dir, seed_exp_name)
            os.makedirs(seed_exp_dir, exist_ok=True)
            current_seed_dirs.append(seed_exp_dir)
            
            # Build command
            cmd = [
                sys.executable, "../main.py",
                "--model", self.config['model'],
                "--mode", "train",
                "--task", self.config['task'],
                "--fold", str(fold),
                "--gpu", str(gpu_id),  # Assign different GPU to each seed
                "--lr", str(self.config['lr_fixed']),
                "--batch_size", str(params_dict['batch_size']),
                "--epochs", str(self.config['epochs']),
                "--patience", str(self.config['patience']),
                "--cxr_encoder", self.config['cxr_encoder'],
                "--hidden_size", str(self.config['hidden_size']),
                "--ehr_dropout", str(self.config['ehr_dropout_fixed']),
                "--ehr_n_head", str(self.config['ehr_n_head']),
                "--ehr_n_layers_distinct", str(self.config['ehr_n_layers_distinct']),
                "--num_prompt_tokens", str(params_dict['num_prompt_tokens']),
                "--prompt_token_dropout", str(params_dict['prompt_token_dropout']),
                "--prompt_noise_std", str(params_dict['prompt_noise_std']),
                "--loss_multi", str(self.config['loss_multi_fixed']),
                "--loss_ehr", str(self.config['loss_ehr_fixed']),
                "--loss_cxr", str(self.config['loss_cxr_fixed']),
                "--aux_ehr_weight", str(self.config['aux_ehr_weight_fixed']),
                "--seed", str(seed),
                "--log_dir", f"/hdd/bayesian_search_experiments/{self.config['model']}/{self.config['task']}"
            ]
            
            # Add fixed parameters
            cmd.extend([
                "--vpt_feature", "hf_chexpert_vit",
                "--hf_model_id", "codewithdark/vit-chest-xray",
                "--freeze_vit", str(self.config['freeze_vit']).lower(),
                "--bias_tune", str(self.config['bias_tune']).lower(),
                "--partial_layers", str(self.config['partial_layers']),
                "--fusion_variant", "null",
                "--final_pred_mode", "concat"
            ])
            
            # Add conditional parameters
            if self.config['matched']:
                cmd.append("--matched")
            if self.config['use_demographics']:
                cmd.append("--use_demographics")
            if self.config['cross_eval']:
                cmd.extend(["--cross_eval", self.config['cross_eval']])
            
            seed_tasks.append((cmd, seed_exp_dir, seed))
        
        # Run all seeds in parallel using threads
        self.log(f"⚡ Running {len(seed_tasks)} seeds in parallel across {num_gpus} GPUs...")
        
        def run_single_seed(cmd, seed_exp_dir, seed):
            """Helper function to run a single seed"""
            try:
                with open(os.path.join(seed_exp_dir, "output.log"), "w") as output_file:
                    result = subprocess.run(
                        cmd,
                        cwd=self.config['base_dir'],
                        stdout=output_file,
                        stderr=subprocess.STDOUT,
                        timeout=None
                    )
                return seed, True, None
            except subprocess.TimeoutExpired:
                return seed, False, "Timeout"
            except Exception as e:
                return seed, False, str(e)
        
        # Execute in parallel using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            # Submit all tasks
            future_to_seed = {
                executor.submit(run_single_seed, cmd, seed_exp_dir, seed): seed 
                for cmd, seed_exp_dir, seed in seed_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.log(f"    📊 Seed {seed} completed")
                except Exception as e:
                    self.log(f"    ❌ Seed {seed} failed: {e}")
                    results.append((seed, False, str(e)))
        
        # Collect metrics from all seeds
        for seed, success, error in results:
            seed_exp_dir = os.path.join(self.results_dir, f"{exp_name}_seed{seed}")
            if success:
                metrics = self.extract_metrics_from_log(os.path.join(seed_exp_dir, "output.log"))
                if metrics and any(v is not None for v in metrics.values()):
                    all_metrics.append(metrics)
                    self.log(f"    ✅ Seed {seed}: " + " | ".join([f"{k}={v:.4f}" if v is not None else f"{k}=N/A" for k, v in metrics.items()]))
                else:
                    self.log(f"    ⚠️  Seed {seed}: Failed to extract metrics")
            else:
                self.log(f"    ❌ Seed {seed}: Error - {error}")
        
        # Calculate statistics for all metrics
        if len(all_metrics) > 0:
            # Calculate mean and std for each metric
            metric_stats = {}
            for metric_name in ['PRAUC', 'ROC_AUC', 'F1_macro', 'F1_weighted', 'Precision_macro', 'Precision_weighted', 'Recall_macro', 'Recall_weighted']:
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
            for metric_name in ['PRAUC', 'ROC_AUC', 'F1_macro', 'F1_weighted', 'Precision_macro', 'Precision_weighted', 'Recall_macro', 'Recall_weighted']:
                mean_key = f'{metric_name}_mean'
                std_key = f'{metric_name}_std'
                if metric_stats[mean_key] is not None:
                    result_parts.append(f"{metric_name}: {metric_stats[mean_key]:.4f}±{metric_stats[std_key]:.4f}")
                else:
                    result_parts.append(f"{metric_name}: N/A")
            
            self.log(result_str + " | ".join(result_parts))
            
            # Update best result based on PRAUC
            prauc_mean = metric_stats.get('PRAUC_mean')
            is_new_best = False
            if prauc_mean is not None and prauc_mean > self.best_score:
                self.best_score = prauc_mean
                self.best_params = params_dict.copy()
                self.best_iteration = self.iteration
                is_new_best = True
                self.log(f"✨ New best PRAUC: {self.best_score:.4f}±{metric_stats.get('PRAUC_std', 0):.4f}")
            
            # Save result
            result_data = {
                'iteration': self.iteration,
                'experiment_name': exp_name,
                'fold': fold,
                'lr_fixed': self.config['lr_fixed'],
                **params_dict,
                'task': self.config['task'],
                'use_demographics': self.config['use_demographics'],
                'cross_eval': self.config['cross_eval'],
                'pretrained': self.config['pretrained'],
                **metric_stats,  # Include all metric statistics
                'all_metrics': all_metrics,  # Include raw metrics from all seeds
                'is_best': is_new_best  # Track if this is the best so far
            }
            self.results_data.append(result_data)
            
            # Strict cleanup policy: only keep checkpoints for the current global-best iteration
            if is_new_best:
                self.log(f"✅ Keeping checkpoints for new best iteration {self.iteration}")
                for previous_result in self.results_data[:-1]:
                    if previous_result['iteration'] != self.best_iteration:
                        self.delete_iteration_checkpoints(
                            previous_result['experiment_name'],
                            iteration_label=previous_result['iteration'],
                        )
            else:
                self.log(f"🗑️  Current iteration {self.iteration} is not best (PRAUC={prauc_mean:.4f}, Best={self.best_score:.4f}); deleting checkpoints")
                self.delete_iteration_checkpoints(exp_name, iteration_label=self.iteration)
            
            return prauc_mean if prauc_mean is not None else -1.0, metric_stats.get('F1_macro_mean', -1.0)
        else:
            self.log(f"Failed to get valid results from any seed in iteration {self.iteration}")
            self.log(f"🗑️  No valid metrics for iteration {self.iteration}; deleting checkpoints")
            self.delete_iteration_checkpoints(exp_name, iteration_label=self.iteration)
            return -1.0, -1.0
    
    def objective_function(self, params):
        """Objective function for Bayesian optimization"""
        # Convert params list to dict
        params_dict = dict(zip(self.dimension_names, params))
        
        # Add fixed parameters that are not in search space
        params_dict['batch_size'] = self.config['batch_size_choices'][0]
        
        # ✨ NEW: Check if this parameter combination has already been run
        params_key = (
            int(params_dict['num_prompt_tokens']),
            float(params_dict['prompt_token_dropout']),
            float(params_dict['prompt_noise_std'])
        )
        
        if params_key in self.already_run_params:
            self.log(f"⏭️  Skipping already-run params: {params_dict}")
            # Return a dummy value (optimization will continue but won't re-run experiment)
            return -0.5  # Return middle-range value to avoid biasing optimization
        
        # Run experiments for all folds and average the results
        scores = []
        for fold in self.config['search_folds']:
            score, _ = self.run_experiment_with_seeds(params_dict, fold)
            scores.append(score)
        
        # Return negative score because skopt minimizes
        avg_score = np.mean(scores)
        return -avg_score  # Negative because we want to maximize PRAUC
    
    def run_optimization(self):
        """Run Bayesian optimization"""
        
        # ✨ NEW: Cleanup-only mode - skip experiments and only cleanup checkpoints
        if self.config.get('cleanup_only', False):
            self.log("✨ CLEANUP-ONLY MODE: Skipping new experiments, only cleaning up checkpoints")
            self.log(f"Loaded {len(self.results_data)} existing experiment(s)")
            self.cleanup_existing_checkpoints_only()
            
            # Save best parameters
            if self.best_params:
                best_params_file = os.path.join(self.results_dir, "best_params.txt")
                with open(best_params_file, 'w') as f:
                    f.write("CrossVPT Bayesian Optimization Best Parameters\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Best PRAUC: {self.best_score:.4f}\n")
                    f.write(f"Total iterations: {len(self.results_data)}\n\n")
                    f.write("Best Parameters:\n")
                    for param, value in self.best_params.items():
                        f.write(f"  {param}: {value}\n")
                self.log(f"Best parameters saved to: {best_params_file}")
            
            # Export best experiment
            self.export_best_experiment()
            return None
        
        self.log("Starting Bayesian Optimization for CrossVPT")
        self.log(f"Fixed Learning Rate: {self.config['lr_fixed']}")
        self.log(f"Fixed Batch Size: {self.config['batch_size_choices']}")
        self.log(f"Fixed EHR Parameters: hidden_size={self.config['hidden_size']}, ehr_dropout={self.config['ehr_dropout_fixed']}, ehr_n_head={self.config['ehr_n_head']}")
        self.log(f"Fixed CXR Encoder: {self.config['cxr_encoder']}")
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
        
        # Save all results to CSV (is_best = True only for the globally best iteration by PRAUC_mean)
        if self.results_data:
            df = pd.DataFrame(self.results_data)
            if 'PRAUC_mean' in df.columns and df['PRAUC_mean'].notna().any():
                best_prauc = df['PRAUC_mean'].max()
                df['is_best'] = (df['PRAUC_mean'] == best_prauc)
            csv_file = os.path.join(self.results_dir, "results_summary.csv")
            df.to_csv(csv_file, index=False)
        
        # Final analysis
        self.log("=== BAYESIAN OPTIMIZATION COMPLETED ===")
        self.log(f"Best PRAUC found: {self.best_score:.4f}")
        self.log(f"Best parameters: {self.best_params}")
        
        # Save best parameters
        best_params_file = os.path.join(self.results_dir, "best_params.txt")
        with open(best_params_file, 'w') as f:
            f.write("CrossVPT Bayesian Optimization Best Parameters\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best PRAUC: {self.best_score:.4f}\n")
            f.write(f"Total iterations: {self.iteration}\n\n")
            f.write("Best Parameters:\n")
            if self.best_params:
                for param, value in self.best_params.items():
                    f.write(f"  {param}: {value}\n")
            f.write(f"Fixed Learning Rate: {self.config['lr_fixed']}\n")
            f.write(f"Fixed Batch Size: {self.config['batch_size_choices']}\n")
            f.write(f"Fixed EHR Parameters: hidden_size={self.config['hidden_size']}, ehr_dropout={self.config['ehr_dropout_fixed']}, ehr_n_head={self.config['ehr_n_head']}\n")
            f.write(f"Fixed CXR Encoder: {self.config['cxr_encoder']}\n")
            f.write(f"Seeds used: {self.config['seeds']}\n")
            if self.best_iteration is not None:
                f.write(f"Best iteration: {self.best_iteration}\n")
        
        # Generate convergence plot
        self.generate_convergence_plot(result)

        # Export the best real training run into a stable BEST_ directory
        self.export_best_experiment()
        
        # ✨ FINAL CHECKPOINT CLEANUP: Keep only best and recent iterations
        self.cleanup_final_checkpoints()
        
        # ✨ NEW: Save detailed metrics for best iteration's three seeds
        self.save_best_iteration_detailed_metrics()
        
        return result
    
    def cleanup_final_checkpoints(self):
        """
        Final cleanup: keep checkpoints only for the best iteration.
        """
        self.log("\n" + "="*60)
        self.log("🧹 FINAL CHECKPOINT CLEANUP")
        self.log("="*60)
        
        if not self.results_data:
            self.log("No results to clean up")
            return

        if self.best_iteration is None:
            self.log("No best iteration found, skipping cleanup")
            return

        self.log(f"✅ Keeping only best iteration: {self.best_iteration}")
        deleted_count = 0

        for result in self.results_data:
            if result['iteration'] != self.best_iteration:
                deleted_count += self.delete_iteration_checkpoints(
                    result['experiment_name'],
                    iteration_label=result['iteration'],
                )

        self.log(f"\n💾 CLEANUP SUMMARY")
        self.log(f"   Deleted {deleted_count} checkpoint directories")
        self.log(f"   Kept checkpoints for best iteration only: {self.best_iteration}")
        self.log("="*60 + "\n")
        
        # ✨ NEW: Save detailed metrics for best iteration's three seeds
        self.save_best_iteration_detailed_metrics()
    
    def generate_convergence_plot(self, result):
        """Generate convergence plot"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10), facecolor='white')
            
            # Plot convergence
            scores = [-y for y in result.func_vals]  # Convert back to positive
            best_scores = [max(scores[:i+1]) for i in range(len(scores))]
            
            plt.subplot(2, 3, 1)
            plt.plot(scores, 'bo-', alpha=0.6, label='PRAUC')
            plt.plot(best_scores, 'r-', linewidth=2, label='Best PRAUC')
            plt.xlabel('Iteration')
            plt.ylabel('PRAUC')
            plt.title('Bayesian Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot parameter exploration
            if hasattr(result, 'x_iters') and len(result.x_iters) > 5:
                param_data = pd.DataFrame(result.x_iters, columns=self.dimension_names)
                
                # Prompt tokens vs performance
                plt.subplot(2, 3, 2)
                plt.scatter(param_data['num_prompt_tokens'], scores, c=scores, cmap='viridis', alpha=0.7)
                plt.colorbar(label='PRAUC')
                plt.xlabel('Number of Prompt Tokens')
                plt.ylabel('PRAUC')
                plt.title('Prompt Tokens vs Performance')
                plt.grid(True, alpha=0.3)
                
                # Prompt dropout progression
                plt.subplot(2, 3, 3)
                plt.plot(param_data['prompt_token_dropout'], 'g-', alpha=0.7)
                plt.xlabel('Iteration')
                plt.ylabel('Prompt Token Dropout')
                plt.title('Prompt Token Dropout Exploration')
                plt.grid(True, alpha=0.3)
                
                # Prompt noise vs performance
                plt.subplot(2, 3, 4)
                plt.scatter(param_data['prompt_noise_std'], scores, c=scores, cmap='viridis', alpha=0.7)
                plt.colorbar(label='PRAUC')
                plt.xlabel('Prompt Noise Std')
                plt.ylabel('PRAUC')
                plt.title('Prompt Noise vs Performance')
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
        'gpus': os.environ.get('GPU', '0'),  # Support multi-GPU (e.g., "0,1,2")
        'epochs': int(os.environ.get('EPOCHS', 50)),
        'patience': int(os.environ.get('PATIENCE', 10)),
        'input_dim': int(os.environ.get('INPUT_DIM', 498)),
        'num_classes': int(os.environ.get('NUM_CLASSES', 25)),
        'pretrained': os.environ.get('PRETRAINED', 'true').lower() == 'true',
        'matched': os.environ.get('MATCHED', 'false').lower() == 'true',
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
        
        # ✨ NEW: Cleanup-only mode
        'cleanup_only': os.environ.get('CLEANUP_ONLY', 'false').lower() == 'true',
        
        # Fixed baseline parameters
        'lr_fixed': float(os.environ.get('LR_FIXED', 0.0001)),
        'batch_size_choices': [int(x) for x in os.environ.get('BATCH_SIZE_CHOICES', '16').split(',')],
        'hidden_size': int(os.environ.get('HIDDEN_SIZE_CHOICES', '256')),
        'ehr_dropout_fixed': float(os.environ.get('EHR_DROPOUT_FIXED', 0.2)),
        'ehr_encoder': os.environ.get('EHR_ENCODER_CHOICES', 'transformer'),
        'cxr_encoder': os.environ.get('CXR_ENCODER_CHOICES', 'hf_chexpert_vit'),
        'freeze_vit': os.environ.get('FREEZE_VIT', 'false').lower() == 'true',
        'bias_tune': os.environ.get('BIAS_TUNE', 'false').lower() == 'true',
        'partial_layers': int(os.environ.get('PARTIAL_LAYERS', 0)),
        'ehr_n_head': int(os.environ.get('EHR_N_HEAD_CHOICES', '4')),
        'ehr_n_layers_distinct': int(os.environ.get('EHR_N_LAYERS_DISTINCT_CHOICES', '1')),
        'seeds': [int(x) for x in os.environ.get('SEEDS', '42,123,1234').split(',')],
        
        # Fixed loss weights (NOT searched)
        'loss_multi_fixed': float(os.environ.get('LOSS_MULTI_FIXED', 1.0)),
        'loss_ehr_fixed': float(os.environ.get('LOSS_EHR_FIXED', 1.0)),
        'loss_cxr_fixed': float(os.environ.get('LOSS_CXR_FIXED', 1.0)),
        'aux_ehr_weight_fixed': float(os.environ.get('AUX_EHR_WEIGHT_FIXED', 1.0)),
        
        # CrossVPT-specific search parameters (ONLY prompt parameters)
        'num_prompt_tokens_choices': [int(x) for x in os.environ.get('NUM_PROMPT_TOKENS_CHOICES', '3,5,7,10').split(',')],
        'prompt_token_dropout_min': float(os.environ.get('PROMPT_TOKEN_DROPOUT_MIN', 0.0)),
        'prompt_token_dropout_max': float(os.environ.get('PROMPT_TOKEN_DROPOUT_MAX', 0.3)),
        'prompt_noise_std_min': float(os.environ.get('PROMPT_NOISE_STD_MIN', 0.0)),
        'prompt_noise_std_max': float(os.environ.get('PROMPT_NOISE_STD_MAX', 0.1)),
    }
    
    # Auto-adjust num_classes for mortality task
    if config['task'] == 'mortality':
        config['num_classes'] = 1
    
    # Create and run optimizer
    optimizer = BayesianCrossVPTOptimizer(config)
    result = optimizer.run_optimization()
    
    print("CrossVPT Bayesian optimization completed successfully!")

if __name__ == "__main__":
    main()
EOF
}

# Main execution
main() {
    if [ "$N_CALLS" -lt "$N_INITIAL_POINTS" ]; then
        log "Adjusting N_INITIAL_POINTS from $N_INITIAL_POINTS to $N_CALLS because skopt requires n_calls >= n_initial_points"
        N_INITIAL_POINTS="$N_CALLS"
    fi

    # ✨ NEW: Handle cleanup-only mode
    if [ "$CLEANUP_ONLY" = "true" ]; then
        log "✨ CLEANUP-ONLY MODE: Will skip new experiments and only cleanup checkpoints"
        log "Scanning for existing experiments in: $RESULTS_DIR"
    else
        log "Starting CrossVPT Bayesian Optimization Search"
        log "Configuration: MODEL=$MODEL, TASK=$TASK, USE_DEMOGRAPHICS=$USE_DEMOGRAPHICS, CROSS_EVAL=$CROSS_EVAL, PRETRAINED=$PRETRAINED"
        log "Results will be saved to: $RESULTS_DIR"
        log "Log file: $LOG_FILE"
        log "Total optimization calls: $N_CALLS"
        log "Initial random points: $N_INITIAL_POINTS"
        log "Acquisition function: $ACQUISITION_FUNC"
    fi
    
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
    
    # ✨ NEW: Cleanup-only mode
    export CLEANUP_ONLY="$CLEANUP_ONLY"
    
    # Fixed baseline parameters
    export LR_FIXED="$LR_FIXED"
    export BATCH_SIZE_CHOICES="$BATCH_SIZE_CHOICES"
    export HIDDEN_SIZE_CHOICES="$HIDDEN_SIZE_CHOICES"
    export EHR_DROPOUT_FIXED="$EHR_DROPOUT_FIXED"
    export EHR_ENCODER_CHOICES="$EHR_ENCODER_CHOICES"
    export CXR_ENCODER_CHOICES="$CXR_ENCODER_CHOICES"
    export FREEZE_VIT="$FREEZE_VIT"
    export BIAS_TUNE="$BIAS_TUNE"
    export PARTIAL_LAYERS="$PARTIAL_LAYERS"
    export EHR_N_HEAD_CHOICES="$EHR_N_HEAD_CHOICES"
    export EHR_N_LAYERS_DISTINCT_CHOICES="$EHR_N_LAYERS_DISTINCT_CHOICES"
    export SEEDS=$(IFS=,; echo "${SEEDS[*]}")
    export LOSS_MULTI_FIXED="$LOSS_MULTI_FIXED"
    export LOSS_EHR_FIXED="$LOSS_EHR_FIXED"
    export LOSS_CXR_FIXED="$LOSS_CXR_FIXED"
    export AUX_EHR_WEIGHT_FIXED="$AUX_EHR_WEIGHT_FIXED"
    
    # CrossVPT-specific search parameters
    export NUM_PROMPT_TOKENS_CHOICES="$NUM_PROMPT_TOKENS_CHOICES"
    export PROMPT_TOKEN_DROPOUT_MIN="$PROMPT_TOKEN_DROPOUT_MIN"
    export PROMPT_TOKEN_DROPOUT_MAX="$PROMPT_TOKEN_DROPOUT_MAX"
    export PROMPT_NOISE_STD_MIN="$PROMPT_NOISE_STD_MIN"
    export PROMPT_NOISE_STD_MAX="$PROMPT_NOISE_STD_MAX"

    log "Starting Python Bayesian optimizer..."
    
    # Run the Bayesian optimizer
    cd "$BASE_DIR"
    python "${RESULTS_DIR}/bayesian_optimizer.py"
    
    if [ $? -eq 0 ]; then
        log "CrossVPT Bayesian optimization completed successfully!"
        log "Results saved to: $RESULTS_DIR"
        log "Best parameters in: $RESULTS_DIR/best_params.txt"
        log "Full results in: $RESULTS_DIR/results_summary.csv"
        log "Optimization object saved in: $RESULTS_DIR/bayesian_optimization_result.pkl"
    else
        log "CrossVPT Bayesian optimization failed!"
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    log "CrossVPT Bayesian search interrupted by user"
    exit 1
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"
