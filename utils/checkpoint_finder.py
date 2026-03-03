"""
Utility functions for finding checkpoints automatically
"""
import os
import re
import glob
from typing import Optional, Dict, Any
from pathlib import Path


def find_experiment_dir(base_dir: str, model: str, task: str, fold: int, seed: int, 
                        matched: bool = True, use_demographics: bool = False,
                        **kwargs) -> Optional[str]:
    """
    Find the experiment directory based on model parameters.
    Uses STRICT matching - all training parameters must match exactly.
    
    Args:
        base_dir: Base directory containing experiments (e.g., './experiments' or './experiments-m-m')
        model: Model name
        task: Task name (phenotype, mortality, los)
        fold: Fold number
        seed: Seed number
        matched: Whether using matched data
        use_demographics: Whether using demographics
        **kwargs: Additional parameters that might be in the directory name
                  (batch_size, lr, patience, epochs, dropout, pretrained, etc.)
        
    Returns:
        Path to experiment directory or None if not found
    """
    # Construct expected directory name pattern
    model_upper = model.upper()
    
    # Determine data config
    if kwargs.get('cross_eval'):
        data_config = f"cross_{kwargs['cross_eval']}"
    elif matched:
        data_config = "matched"
    else:
        data_config = "full"
    
    if use_demographics:
        data_config += "_demo"
    
    # Search in lightning_logs subdirectory
    search_pattern = os.path.join(base_dir, model, task, "lightning_logs", "*")
    
    # Find matching directories
    potential_dirs = glob.glob(search_pattern)
    
    # Build list of required parameter patterns to match
    # CRITICAL: Use word boundaries or exact matching for seed to avoid matching seed_123 with seed_1234
    seed_pattern = f"seed_{seed}"  # Will be checked with word boundary logic
    
    required_patterns = [
        f"task_{task}",
        f"fold_{fold}",
        model_upper,
        f"data_config_{data_config}"
    ]
    
    # Add training parameter patterns if provided in kwargs
    # These are critical for exact matching
    # Note: dropout is optional - some models don't have it in directory name
    training_params = ['batch_size', 'lr', 'patience', 'epochs']
    for param in training_params:
        if param in kwargs and kwargs[param] is not None:
            value = kwargs[param]
            required_patterns.append(f"{param}_{value}")
    
    # Handle dropout separately - only add if provided and not None
    # Some models (like latefusion, drfuse) don't have dropout in directory name
    if 'dropout' in kwargs and kwargs['dropout'] is not None:
        dropout_value = kwargs['dropout']
        dropout_pattern = f"dropout_{dropout_value}"
    else:
        dropout_pattern = None
    
    # Handle pretrained separately - only check if directory name contains it
    # Some models don't have pretrained in directory name (e.g., transformer, umse, utde)
    pretrained_value = kwargs.get('pretrained', None)
    pretrained_pattern = f"pretrained_{str(pretrained_value)}" if pretrained_value is not None else None
    
    # Only allow exact match - all required patterns must be present
    for dir_path in potential_dirs:
        dir_name = os.path.basename(dir_path)
        
        # Check seed with exact matching (not substring matching)
        # seed_123 should NOT match seed_1234
        # Use word boundary: seed_123 should be followed by - or end of string
        seed_match = False
        seed_pos = dir_name.find(seed_pattern)
        if seed_pos != -1:
            # Check if it's followed by a delimiter (-) or end of string
            next_char_pos = seed_pos + len(seed_pattern)
            if next_char_pos >= len(dir_name) or dir_name[next_char_pos] == '-':
                seed_match = True
        
        if not seed_match:
            continue  # Skip this directory if seed doesn't match exactly
        
        # Check if ALL other required patterns are present
        all_match = all(pattern in dir_name for pattern in required_patterns)
        
        # Check dropout pattern if specified
        # Some models don't have dropout in directory name, so we only check if it's there
        if all_match and dropout_pattern:
            # If directory name contains dropout, it must match
            if 'dropout' in dir_name:
                if dropout_pattern not in dir_name:
                    all_match = False
            # If directory name doesn't contain dropout, and we're looking for dropout,
            # that's OK - just skip the dropout check (some models don't use it)
            # So we don't reject the directory
        
        # If pretrained pattern is specified, check if it matches (if present in dir_name)
        # Some models don't have pretrained in directory name, so we only check if it's there
        if all_match and pretrained_pattern:
            # If directory name contains pretrained, it must match
            if 'pretrained' in dir_name:
                if pretrained_pattern not in dir_name:
                    all_match = False
            # If directory name doesn't contain pretrained, and we're looking for pretrained_True,
            # that's OK - just skip the pretrained check (some models don't use it)
            # So we don't reject the directory
        
        if all_match:
            return dir_path
    
    # No exact match found - return None (strict matching)
    return None


def find_best_checkpoint(experiment_dir: str, task: str = 'phenotype') -> Optional[str]:
    """
    Find the best checkpoint in an experiment directory based on metric value in filename.
    
    Args:
        experiment_dir: Path to experiment directory (should contain checkpoints/ subdirectory)
        task: Task name to determine which metric to use (phenotype/mortality use PRAUC, los uses ACC)
        
    Returns:
        Path to best checkpoint file or None if not found
    """
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    
    # Determine metric name based on task
    if task == 'los':
        metric_name = 'ACC'
        # Support multiple formats: ACC=0.1234.ckpt or acc_0.1234.ckpt or ACC_0.1234.ckpt
        metric_patterns = [
            r'ACC=([0-9.]+)\.ckpt',
            r'acc[_\s]+([0-9.]+)\.ckpt',
            r'ACC[_\s]+([0-9.]+)\.ckpt'
        ]
    else:
        metric_name = 'PRAUC'
        # Support multiple formats: PRAUC=0.1234.ckpt or prauc_0.1234.ckpt or PRAUC_0.1234.ckpt
        metric_patterns = [
            r'PRAUC=([0-9.]+)\.ckpt',
            r'prauc[_\s]+([0-9.]+)\.ckpt',
            r'PRAUC[_\s]+([0-9.]+)\.ckpt'
        ]
    
    # First try the standard checkpoints/ subdirectory
    checkpoint_files = []
    if os.path.exists(checkpoints_dir):
        for root, dirs, files in os.walk(checkpoints_dir):
            for file in files:
                if file.endswith('.ckpt'):
                    file_path = os.path.join(root, file)
                    # Try to extract metric value from filename using multiple patterns
                    metric_value = None
                    for pattern in metric_patterns:
                        match = re.search(pattern, file, re.IGNORECASE)
                        if match:
                            try:
                                metric_value = float(match.group(1))
                                break
                            except ValueError:
                                continue
                    if metric_value is None:
                        # If no pattern matches, still include it with value 0 (fallback)
                        metric_value = 0.0
                    checkpoint_files.append((metric_value, file_path))
    
    # If no checkpoints found in checkpoints/ directory, search in experiment directory itself
    if not checkpoint_files:
        print(f"Warning: checkpoints/ directory not found in {experiment_dir}")
        print(f"Searching for checkpoint files in the experiment directory...")
        
        # Search for .ckpt files in the experiment directory itself
        for file in os.listdir(experiment_dir):
            if file.endswith('.ckpt'):
                file_path = os.path.join(experiment_dir, file)
                # Try to extract metric value from filename using multiple patterns
                metric_value = None
                for pattern in metric_patterns:
                    match = re.search(pattern, file, re.IGNORECASE)
                    if match:
                        try:
                            metric_value = float(match.group(1))
                            break
                        except ValueError:
                            continue
                if metric_value is None:
                    # If no pattern matches, still include it with value 0 (fallback)
                    metric_value = 0.0
                checkpoint_files.append((metric_value, file_path))
    
    if not checkpoint_files:
        print(f"Error: No checkpoint files found in {experiment_dir}")
        if os.path.exists(experiment_dir):
            available_files = [f for f in os.listdir(experiment_dir) if not f.startswith('.')]
            print(f"Available files in experiment directory: {available_files[:10]}")
        return None
    
    # Sort by metric value (descending) and return the best one
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    best_checkpoint = checkpoint_files[0][1]
    
    print(f"Found {len(checkpoint_files)} checkpoint(s), best {metric_name}: {checkpoint_files[0][0]:.4f}")
    print(f"Using checkpoint: {best_checkpoint}")
    
    return best_checkpoint


def auto_find_checkpoint(base_dir: str, model: str, task: str, fold: int, seed: int,
                         matched: bool = True, use_demographics: bool = False,
                         **kwargs) -> Optional[str]:
    """
    Automatically find the best checkpoint for given experiment parameters.
    
    Args:
        base_dir: Base directory containing experiments
        model: Model name
        task: Task name
        fold: Fold number
        seed: Seed number
        matched: Whether using matched data
        use_demographics: Whether using demographics
        **kwargs: Additional parameters
        
    Returns:
        Path to best checkpoint or None if not found
    """
    # Find experiment directory
    experiment_dir = find_experiment_dir(
        base_dir, model, task, fold, seed, matched, use_demographics, **kwargs
    )
    
    if experiment_dir is None:
        print(f"Could not find experiment directory for model={model}, task={task}, fold={fold}, seed={seed}")
        print(f"  Searched in: {base_dir}")
        print(f"  Expected data_config: {'matched' if matched else 'full'}")
        return None
    
    print(f"Found experiment directory: {experiment_dir}")
    
    # Find best checkpoint
    checkpoint_path = find_best_checkpoint(experiment_dir, task)
    
    return checkpoint_path
