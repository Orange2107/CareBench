#!/bin/bash

# ============================================================================
# HEALNET BAYESIAN OPTIMIZATION SEARCH - ALIGN WITH CROSSVPT SETTINGS
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
#   CLEANUP_ONLY=true ./bayesian_search_healnet_hf.sh
#
#   # Or edit this file and set: CLEANUP_ONLY=true

'''
python main.py \
  --model healnet \
  --mode train \
  --task phenotype \
  --fold 1 \
  --gpu 0 \
  --lr 0.0001 \
  --batch_size 16 \
  --epochs 50 \
  --patience 10 \
  --cxr_encoder hf_chexpert_vit \
  --input_dim 49 \
  --num_classes 9 \
  --use_phenotype9 \
  --n_modalities 2 \
  --latent_channels 256 \
  --latent_dim 256 \
  --cross_heads 4 \
  --latent_heads 4 \
  --cross_dim_head 64 \
  --latent_dim_head 64 \
  --self_per_cross_attn 1 \
  --attn_dropout 0.2 \
  --ff_dropout 0.2 \
  --dropout 0.2 \
  --depth 2 \
  --num_freq_bands 2 \
  --max_freq 10.0 \
  --seed 42 \
  --log_dir /hdd/bayesian_search_experiments/healnet/phenotype
'''


# Fold selection for bayesian search (can modify to include more folds)
SEARCH_FOLDS=(1)  

# Model Configuration
MODEL="healnet"
TASK="phenotype"  # phenotype, mortality, or los (aligned with baselines)
GPU="0,1,2 "  # 支持多 GPU 并行，用逗号分隔（如 "0,1,2" 表示 3 个 GPU）

# Basic Experiment Settings
PRETRAINED=true
USE_DEMOGRAPHICS=false
CROSS_EVAL=""  # Set to "matched_to_full" or "full_to_matched" if needed
MATCHED=false  # Aligned with baselines (full data, not matched)
INPUT_DIM=49   # EHR input dimension for HealNet

# Bayesian Optimization Settings
N_CALLS=20                    # Total number of optimization iterations
N_INITIAL_POINTS=5            # Number of random initial points
ACQUISITION_FUNC="gp_hedge"   # Acquisition function: 'LCB', 'EI', 'PI', 'gp_hedge'
N_JOBS=8                      # Number of parallel jobs (-1 for all cores)

# Resume settings
RESUME_FROM_CHECKPOINT=true  # Set to true to resume from previous run
CHECKPOINT_FILE=""            # Path to previous bayesian_optimization_result.pkl (auto-detect if empty)

# ✨ NEW: Cleanup-only mode (skip experiments, only cleanup checkpoints)
CLEANUP_ONLY=false          # Set to true to only cleanup checkpoints without running new experiments

# Search Space Bounds - Define parameter ranges for Bayesian optimization
# FIXED PARAMETERS (same as other baselines):
# - batch_size: 16
# - learning_rate: 0.0001
# - cxr_encoder: hf_chexpert_vit
# - input_dim: 49 (HealNet specific)

# Core training parameters - FIXED
LR_FIXED=0.0001              # Fixed learning rate (same as baselines)
BATCH_SIZE_CHOICES="16"      # Fixed batch size (same as baselines)
EPOCHS_VALUES=(50)           # Keep epochs reasonable
PATIENCE_VALUES=(10)         # Patience for early stopping

# Seeds for multiple runs
SEEDS=(42 123 1234)

# Task-specific parameters
NUM_CLASSES_VALUES=(9)       # For phenotype: 9 (Phenotype9 dataset)

# HealNet architecture parameters - FIXED (aligned with baselines where applicable)
CXR_ENCODER_CHOICES="hf_chexpert_vit"  # HF ViT encoder (fixed, aligned with baselines)
N_MODALITIES_FIXED=2                  # Fixed number of modalities (EHR and CXR)
LATENT_CHANNELS_FIXED=256             # Fixed latent channels (from config)
LATENT_DIM_FIXED=256                  # Fixed latent dimension (from config)
CROSS_HEADS_FIXED=4                   # Fixed cross-attention heads (from config)
LATENT_HEADS_FIXED=4                  # Fixed self-attention heads (from config)
CROSS_DIM_HEAD_FIXED=64               # Fixed cross-attention head dimension (from config)
LATENT_DIM_HEAD_FIXED=64              # Fixed self-attention head dimension (from config)
SELF_PER_CROSS_ATTN_FIXED=1           # Fixed self-attention per cross-attention (from config)
WEIGHT_TIE_LAYERS_FIXED=true          # Fixed weight tying (from config)
SNN_FIXED=true                        # Fixed self-normalizing networks (from config)
FOURIER_ENCODE_DATA_FIXED=true        # Fixed Fourier encoding (from config)
FINAL_CLASSIFIER_HEAD_FIXED=true      # Fixed final classifier head (from config)
ATTN_DROPOUT_FIXED=0.2                # Fixed attention dropout (from config)
FF_DROPOUT_FIXED=0.2                  # Fixed feed-forward dropout (from config)
DROPOUT_FIXED=0.2                     # Fixed general dropout (from config)

# SEARCH PARAMETERS: HealNet architecture parameters (ONLY THESE ARE SEARCHED!)
# Based on config comments: depth: 1、2、3; num_freq_bands: 1、2、4; max_freq: 5、10.0
DEPTH_CHOICES="1,2,3"                 # Fusion depth choices
NUM_FREQ_BANDS_CHOICES="1,2,4"        # Frequency bands choices
MAX_FREQ_CHOICES="5.0,10.0"           # Maximum frequency choices

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

class BayesianHealNetOptimizer:
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
        
        # Define search space for HealNet - ONLY architecture parameters
        self.dimensions = [
            Categorical(config['depth_choices'], name='depth'),
            Categorical(config['num_freq_bands_choices'], name='num_freq_bands'),
            Categorical(config['max_freq_choices'], name='max_freq')
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
        
        # Method 1: Look for results_summary.csv
        csv_file = os.path.join(self.results_dir, "results_summary.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    # Extract parameter tuple: (depth, num_freq_bands, max_freq)
                    params_key = (
                        int(row['depth']),
                        int(row['num_freq_bands']),
                        float(row['max_freq'])
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
                        'depth': int(best_row['depth']),
                        'num_freq_bands': int(best_row['num_freq_bands']),
                        'max_freq': float(best_row['max_freq']),
                        'batch_size': int(best_row.get('batch_size', 16))
                    }
                    self.log(f"✨ Best existing result: Iteration {self.best_iteration}, PRAUC={self.best_score:.4f}")
                    self.log(f"   Best params: {self.best_params}")
                    
            except Exception as e:
                self.log(f"⚠️  Failed to read results_summary.csv: {e}")
        
        # Method 2: ✨ NEW: Scan experiment directories and check test_set_results.yaml
        # IMPORTANT: Experiment directories are in the parent directory (lightning_logs), not in results_dir
        self.log("🔍 Scanning experiment directories for completed runs...")
        completed_count = 0
        
        # Scan parent directory where experiment directories are located
        parent_dir = os.path.dirname(self.results_dir)
        self.log(f"   Scanning directory: {parent_dir}")
        
        # Pattern to match experiment directories: healnet_fold1_d{depth}_fb{num_freq_bands}_mf{max_freq}_seed{seed}
        exp_pattern = re.compile(
            r'healnet_fold\d+_d(\d+)_fb(\d+)_mf([\d.]+)_seed(\d+)'
        )
        
        for entry in os.listdir(parent_dir):
            entry_path = os.path.join(parent_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            
            match = exp_pattern.match(entry)
            if not match:
                continue
            
            depth = int(match.group(1))
            num_freq_bands = int(match.group(2))
            max_freq = float(match.group(3))
            seed = int(match.group(4))
            
            # Check if test_set_results.yaml exists for this seed
            results_yaml = os.path.join(entry_path, 'test_set_results.yaml')
            if os.path.exists(results_yaml):
                # This seed run is completed - just check file existence
                params_key = (depth, num_freq_bands, max_freq)
                self.already_run_params.add(params_key)
                completed_count += 1
                self.log(f"   ✅ Found completed: d{depth}_fb{num_freq_bands}_mf{max_freq}_seed{seed}")
            else:
                self.log(f"   ⏳ Incomplete (no test_set_results.yaml): {entry}")
        
        self.log(f"✅ Found {completed_count} completed experiment(s) by checking test_set_results.yaml")
        self.log(f"📊 Total unique parameter combinations to skip: {len(self.already_run_params)}")
    
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
        best_depth = int(best_result['depth'])
        best_freq_bands = int(best_result['num_freq_bands'])
        best_max_freq = float(best_result['max_freq'])
        
        self.log(f"Best params: depth={best_depth}, freq_bands={best_freq_bands}, max_freq={best_max_freq}")
        
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
                    value = seed_metrics[metric_key]
                    # Convert to float if possible, skip non-numeric values
                    try:
                        if value is not None:
                            numeric_value = float(value)
                            values.append(numeric_value)
                    except (ValueError, TypeError):
                        # Skip non-numeric values (strings, None, etc.)
                        pass
            
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
            self.log(f"\n📊 Preview (overall metric):")
            if 'overall/PRAUC' in all_metric_keys:
                prauc_row = next((r for r in detailed_results if r['metric'] == 'overall/PRAUC'), None)
                if prauc_row:
                    self.log(f"   overall/PRAUC: {prauc_row['formatted']}")
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
        exact_dir = os.path.join(self.results_dir, f"{exp_name}_seed{seed}")
        if os.path.isdir(exact_dir):
            return exact_dir
        return None

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

    def _extract_search_params_from_exp_name(self, exp_name):
        """
        Parse depth/num_freq_bands/max_freq from either old or new experiment name formats.
        Supports:
          - Old: depth3_freqbands1_maxfreq10.0
          - Legacy: _depth3_nfb1_mf10.0
          - New: d3_fb1_mf10.0 (aligned with utils/ver_name.py)
        """
        # Old format: depth3_freqbands1_maxfreq10.0
        old_match = re.search(r'depth(\d+)_freqbands(\d+)_maxfreq([0-9.]+)', exp_name)
        if old_match:
            return old_match.group(1), old_match.group(2), old_match.group(3)
        
        # Legacy format: _depth3_nfb1_mf10.0
        new_match = re.search(r'_depth(\d+)_nfb(\d+)_mf([0-9.]+)', exp_name)
        if new_match:
            return new_match.group(1), new_match.group(2), new_match.group(3)
        
        # New format: d3_fb1_mf10.0 (aligned with utils/ver_name.py)
        latest_match = re.search(r'_d(\d+)_fb(\d+)_mf([0-9.]+)', exp_name)
        if latest_match:
            return latest_match.group(1), latest_match.group(2), latest_match.group(3)
        
        return None, None, None

    def _format_float_tag(self, value):
        return f"{float(value):.4g}"

    def _seed_match(self, dir_name, seed):
        return (f"seed_{seed}" in dir_name) or (f"seed{seed}" in dir_name)

    def _depth_match(self, dir_name, depth):
        return (f"depth_{depth}" in dir_name) or (f"depth{depth}" in dir_name)

    def _nfb_match(self, dir_name, freq_bands):
        return (
            (f"num_freq_bands_{freq_bands}" in dir_name) or
            (f"freqbands{freq_bands}" in dir_name) or
            (f"nfb{freq_bands}" in dir_name)
        )

    def _max_freq_match(self, dir_name, max_freq):
        max_freq_fmt = self._format_float_tag(max_freq)
        return (
            (f"max_freq_{max_freq}" in dir_name) or
            (f"max_freq_{max_freq_fmt}" in dir_name) or
            (f"maxfreq{max_freq}" in dir_name) or
            (f"maxfreq{max_freq_fmt}" in dir_name) or
            (f"mf{max_freq}" in dir_name) or
            (f"mf{max_freq_fmt}" in dir_name)
        )

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
        depth = result_row.get('depth')
        num_freq_bands = result_row.get('num_freq_bands')
        max_freq = result_row.get('max_freq')
        fold = result_row.get('fold')
        prauc = result_row.get('PRAUC_mean')

        return (
            f"BEST_{self.config['model']}_{self.config['task']}_"
            f"fold{fold}_"
            f"depth{depth}_"
            f"freqbands{num_freq_bands}_"
            f"maxfreq{max_freq}_"
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
        
        # Convert numpy types to Python native types for JSON serialization
        metadata = {
            'best_iteration': int(self.best_iteration) if hasattr(self.best_iteration, 'item') else self.best_iteration,
            'best_experiment_name': best_row['experiment_name'],
            'best_score': float(self.best_score) if hasattr(self.best_score, 'item') else self.best_score,
            'exported_seeds': exported_seeds,
            'best_params': {k: (float(v) if hasattr(v, 'item') else v) for k, v in self.best_params.items()},
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
                self.log(f"   ⚠️  No matching directories found for iter {iteration_label} seed {seed}")
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
        
        # Create base experiment name (aligned with utils/ver_name.py template)
        # Format: {model}_fold{fold}_d{depth}_fb{num_freq_bands}_mf{max_freq}_seed{seeds}
        exp_base = (
            f"{self.config['model']}_fold{fold}_"
            f"d{params_dict['depth']}_fb{params_dict['num_freq_bands']}_"
            f"mf{self._format_float_tag(params_dict['max_freq'])}"
        )
        
        # ✨ NEW: Check if this parameter combination has already been completed
        params_key = (
            int(params_dict['depth']),
            int(params_dict['num_freq_bands']),
            float(params_dict['max_freq'])
        )
        
        if params_key in self.already_run_params:
            self.log(f"⏭️  SKIP iteration {self.iteration + 1}: {exp_base} (already completed)")
            
            # Try to load existing results from CSV to get the score
            csv_file = os.path.join(self.results_dir, "results_summary.csv")
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    match_row = df[
                        (df['depth'] == params_dict['depth']) &
                        (df['num_freq_bands'] == params_dict['num_freq_bands']) &
                        (df['max_freq'] == params_dict['max_freq'])
                    ]
                    if len(match_row) > 0 and 'PRAUC_mean' in match_row.iloc[0]:
                        prauc_mean = match_row.iloc[0]['PRAUC_mean']
                        self.log(f"   📊 Existing result: PRAUC_mean = {prauc_mean:.4f}")
                        if pd.notna(prauc_mean):
                            return prauc_mean, None
                except Exception as e:
                    self.log(f"   ⚠️  Could not load existing result: {e}")
            
            # Return a dummy value to continue optimization
            return -1.0, None
        
        # This is a new parameter combination, increment iteration counter
        self.iteration += 1
        self.log(f"Starting Bayesian iteration {self.iteration}: {exp_base}")
        
        # Parse available GPUs
        gpu_list = [int(g) for g in str(self.config['gpus']).split(',')]
        num_gpus = len(gpu_list)
        self.log(f"🚀 Using {num_gpus} GPUs for parallel execution: {gpu_list}")
        
        # Track experiment directories for potential cleanup
        current_seed_dirs = []
        
        # Initialize metrics collection
        all_metrics = []
        
        # ✨ MULTI-GPU PARALLEL EXECUTION using subprocess (no multiprocessing)
        
        # Prepare tasks for all seeds - each seed gets its own directory
        seed_tasks = []
        for i, seed in enumerate(self.config['seeds']):
            gpu_id = gpu_list[i % num_gpus]  # Round-robin GPU assignment
            seed_exp_name = f"{exp_base}_seed{seed}"
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
                "--input_dim", str(self.config['input_dim']),
                "--num_classes", str(self.config['num_classes']),
                "--cxr_encoder", self.config['cxr_encoder'],
                "--n_modalities", str(self.config['n_modalities_fixed']),
                "--latent_channels", str(self.config['latent_channels_fixed']),
                "--latent_dim", str(self.config['latent_dim_fixed']),
                "--cross_heads", str(self.config['cross_heads_fixed']),
                "--latent_heads", str(self.config['latent_heads_fixed']),
                "--cross_dim_head", str(self.config['cross_dim_head_fixed']),
                "--latent_dim_head", str(self.config['latent_dim_head_fixed']),
                "--self_per_cross_attn", str(self.config['self_per_cross_attn_fixed']),
                "--attn_dropout", str(self.config['attn_dropout_fixed']),
                "--ff_dropout", str(self.config['ff_dropout_fixed']),
                "--dropout", str(self.config['dropout_fixed']),
                "--depth", str(params_dict['depth']),
                "--num_freq_bands", str(params_dict['num_freq_bands']),
                "--max_freq", str(params_dict['max_freq']),
                "--seed", str(seed),
                "--log_dir", f"/hdd/bayesian_search_experiments/{self.config['model']}/{self.config['task']}"
            ]
            
            # Add conditional parameters
            if self.config['pretrained']:
                cmd.append("--pretrained")
            if self.config['matched']:
                cmd.append("--matched")
            if self.config['use_demographics']:
                cmd.append("--use_demographics")
            if self.config['weight_tie_layers_fixed']:
                cmd.append("--weight_tie_layers")
            if self.config['snn_fixed']:
                cmd.append("--snn")
            if self.config['fourier_encode_data_fixed']:
                cmd.append("--fourier_encode_data")
            if self.config['final_classifier_head_fixed']:
                cmd.append("--final_classifier_head")
            if self.config['cross_eval']:
                cmd.extend(["--cross_eval", self.config['cross_eval']])
            
            # Add Phenotype9 dataset flag
            if self.config['num_classes'] == 9:
                cmd.append("--use_phenotype9")
            
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
        
        # Collect metrics from all seeds
        for seed, success, error in results:
            seed_exp_dir = os.path.join(self.results_dir, f"{exp_base}_seed{seed}")
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
                'experiment_name': exp_base,
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
                self.delete_iteration_checkpoints(exp_base, iteration_label=self.iteration)
            
            return prauc_mean if prauc_mean is not None else -1.0, metric_stats.get('F1_macro_mean', -1.0)
        else:
            self.log(f"Failed to get valid results from any seed in iteration {self.iteration}")
            self.log(f"🗑️  No valid metrics for iteration {self.iteration}; deleting checkpoints")
            self.delete_iteration_checkpoints(exp_base, iteration_label=self.iteration)
            return -1.0, -1.0
    
    def objective_function(self, params):
        """Objective function for Bayesian optimization"""
        # Convert params list to dict
        params_dict = dict(zip(self.dimension_names, params))
        
        # Add fixed parameters that are not in search space
        params_dict['batch_size'] = self.config['batch_size_choices'][0]
        
        # ✨ NEW: Check if this parameter combination has already been run
        params_key = (
            int(params_dict['depth']),
            int(params_dict['num_freq_bands']),
            float(params_dict['max_freq'])
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
                    f.write("HealNet Bayesian Optimization Best Parameters\n")
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
        
        self.log("Starting Bayesian Optimization for HealNet (architecture parameters)")
        self.log(f"Fixed Learning Rate: {self.config['lr_fixed']}")
        self.log(f"Fixed Batch Size: {self.config['batch_size_choices']}")
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
            f.write("HealNet Bayesian Optimization Best Parameters\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best PRAUC: {self.best_score:.4f}\n")
            f.write(f"Total iterations: {self.iteration}\n\n")
            f.write("Best Parameters:\n")
            if self.best_params:
                for param, value in self.best_params.items():
                    f.write(f"  {param}: {value}\n")
            f.write(f"Fixed Learning Rate: {self.config['lr_fixed']}\n")
            f.write(f"Fixed Batch Size: {self.config['batch_size_choices']}\n")
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
                
                # Depth vs performance
                plt.subplot(2, 3, 2)
                plt.scatter(param_data['depth'], scores, c=scores, cmap='viridis', alpha=0.7)
                plt.colorbar(label='PRAUC')
                plt.xlabel('Depth')
                plt.ylabel('PRAUC')
                plt.title('Depth vs Performance')
                plt.grid(True, alpha=0.3)
                
                # Frequency bands progression
                plt.subplot(2, 3, 3)
                plt.plot(param_data['num_freq_bands'], 'g-', alpha=0.7)
                plt.xlabel('Iteration')
                plt.ylabel('Number of Frequency Bands')
                plt.title('Frequency Bands Exploration')
                plt.grid(True, alpha=0.3)
                
                # Max frequency vs performance
                plt.subplot(2, 3, 4)
                plt.scatter(param_data['max_freq'], scores, c=scores, cmap='viridis', alpha=0.7)
                plt.colorbar(label='PRAUC')
                plt.xlabel('Max Frequency')
                plt.ylabel('PRAUC')
                plt.title('Max Frequency vs Performance')
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
        'input_dim': int(os.environ.get('INPUT_DIM', 49)),
        'num_classes': int(os.environ.get('NUM_CLASSES', 9)),
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
        'cxr_encoder': os.environ.get('CXR_ENCODER_CHOICES', 'hf_chexpert_vit'),
        'seeds': [int(x) for x in os.environ.get('SEEDS', '42,123,1234').split(',')],
        
        # HealNet-specific fixed parameters
        'n_modalities_fixed': int(os.environ.get('N_MODALITIES_FIXED', 2)),
        'latent_channels_fixed': int(os.environ.get('LATENT_CHANNELS_FIXED', 256)),
        'latent_dim_fixed': int(os.environ.get('LATENT_DIM_FIXED', 256)),
        'cross_heads_fixed': int(os.environ.get('CROSS_HEADS_FIXED', 4)),
        'latent_heads_fixed': int(os.environ.get('LATENT_HEADS_FIXED', 4)),
        'cross_dim_head_fixed': int(os.environ.get('CROSS_DIM_HEAD_FIXED', 64)),
        'latent_dim_head_fixed': int(os.environ.get('LATENT_DIM_HEAD_FIXED', 64)),
        'self_per_cross_attn_fixed': int(os.environ.get('SELF_PER_CROSS_ATTN_FIXED', 1)),
        'weight_tie_layers_fixed': os.environ.get('WEIGHT_TIE_LAYERS_FIXED', 'true').lower() == 'true',
        'snn_fixed': os.environ.get('SNN_FIXED', 'true').lower() == 'true',
        'fourier_encode_data_fixed': os.environ.get('FOURIER_ENCODE_DATA_FIXED', 'true').lower() == 'true',
        'final_classifier_head_fixed': os.environ.get('FINAL_CLASSIFIER_HEAD_FIXED', 'true').lower() == 'true',
        'attn_dropout_fixed': float(os.environ.get('ATTN_DROPOUT_FIXED', 0.2)),
        'ff_dropout_fixed': float(os.environ.get('FF_DROPOUT_FIXED', 0.2)),
        'dropout_fixed': float(os.environ.get('DROPOUT_FIXED', 0.2)),
        
        # Search parameter choices
        'depth_choices': [int(x) for x in os.environ.get('DEPTH_CHOICES', '1,2,3').split(',')],
        'num_freq_bands_choices': [int(x) for x in os.environ.get('NUM_FREQ_BANDS_CHOICES', '1,2,4').split(',')],
        'max_freq_choices': [float(x) for x in os.environ.get('MAX_FREQ_CHOICES', '5.0,10.0').split(',')]
    }
    
    # Auto-adjust num_classes for mortality task
    if config['task'] == 'mortality':
        config['num_classes'] = 1
    
    # Create and run optimizer
    optimizer = BayesianHealNetOptimizer(config)
    result = optimizer.run_optimization()
    
    print("HealNet Bayesian optimization completed successfully!")

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
        log "Starting HealNet Bayesian Optimization Search (aligned with CrossVPT settings)"
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
    export CXR_ENCODER_CHOICES="$CXR_ENCODER_CHOICES"
    export SEEDS=$(IFS=,; echo "${SEEDS[*]}")
    
    # HealNet-specific fixed parameters
    export N_MODALITIES_FIXED="$N_MODALITIES_FIXED"
    export LATENT_CHANNELS_FIXED="$LATENT_CHANNELS_FIXED"
    export LATENT_DIM_FIXED="$LATENT_DIM_FIXED"
    export CROSS_HEADS_FIXED="$CROSS_HEADS_FIXED"
    export LATENT_HEADS_FIXED="$LATENT_HEADS_FIXED"
    export CROSS_DIM_HEAD_FIXED="$CROSS_DIM_HEAD_FIXED"
    export LATENT_DIM_HEAD_FIXED="$LATENT_DIM_HEAD_FIXED"
    export SELF_PER_CROSS_ATTN_FIXED="$SELF_PER_CROSS_ATTN_FIXED"
    export WEIGHT_TIE_LAYERS_FIXED="$WEIGHT_TIE_LAYERS_FIXED"
    export SNN_FIXED="$SNN_FIXED"
    export FOURIER_ENCODE_DATA_FIXED="$FOURIER_ENCODE_DATA_FIXED"
    export FINAL_CLASSIFIER_HEAD_FIXED="$FINAL_CLASSIFIER_HEAD_FIXED"
    export ATTN_DROPOUT_FIXED="$ATTN_DROPOUT_FIXED"
    export FF_DROPOUT_FIXED="$FF_DROPOUT_FIXED"
    export DROPOUT_FIXED="$DROPOUT_FIXED"
    
    # Search parameter choices
    export DEPTH_CHOICES="$DEPTH_CHOICES"
    export NUM_FREQ_BANDS_CHOICES="$NUM_FREQ_BANDS_CHOICES"
    export MAX_FREQ_CHOICES="$MAX_FREQ_CHOICES"

    log "Starting Python Bayesian optimizer..."
    
    # Run the Bayesian optimizer
    cd "$BASE_DIR"
    python "${RESULTS_DIR}/bayesian_optimizer.py"
    
    if [ $? -eq 0 ]; then
        log "HealNet Bayesian optimization completed successfully!"
        log "Results saved to: $RESULTS_DIR"
        log "Best parameters in: $RESULTS_DIR/best_params.txt"
        log "Full results in: $RESULTS_DIR/results_summary.csv"
        log "Optimization object saved in: $RESULTS_DIR/bayesian_optimization_result.pkl"
    else
        log "HealNet Bayesian optimization failed!"
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    log "HealNet Bayesian search interrupted by user"
    exit 1
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"
