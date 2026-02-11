"""
Argument Parsing
----------------
Command-line args + YAML config support.
"""

import argparse
import yaml
import os
from typing import Dict, Any
from argparse import Namespace


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML file as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_args() -> Namespace:
    """
    Parse args with YAML config support.
    Priority: CLI > --config_path > configs/{model}.yaml > defaults.
    """
    parser = argparse.ArgumentParser(description='Unified Clinical Learner Args')

    # Core arguments
    parser.add_argument('--model', type=str, required=True, help='Model name (finds YAML config)')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='GPU index/indices')
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, nargs='+', default=[42], 
                       help='Random seed(s). Can specify multiple seeds: --seed 42 123 1234')
    parser.add_argument('--save_checkpoint', action='store_true', default=True)
    parser.add_argument('--dev_run', action='store_true')
    parser.add_argument('--use_triplet', action='store_true')
    parser.add_argument('--config_root', type=str, default=os.path.join(os.path.dirname(__file__), 'configs'))
    parser.add_argument('--config_path', type=str, default=None, help='Custom YAML config path')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint for test')

    # Data splitting
    parser.add_argument('--matched', action='store_true', help='Use matched subset')
    parser.add_argument('--cross_eval', type=str, choices=['matched_to_full', 'full_to_matched'],
                       help='Cross evaluation mode')

    # Demographic features
    parser.add_argument('--use_demographics', action='store_true', help='Use demographic features')
    parser.add_argument('--demographic_cols', type=str, nargs='+',
                       default=['age', 'gender', 'admission_type', 'race'],
                       help='Demographic columns to use')

    # Label weights for imbalance
    parser.add_argument('--use_label_weights', action='store_true', help='Use label weights')
    parser.add_argument('--label_weight_method', type=str,
                       choices=['balanced', 'inverse', 'sqrt_inverse', 'log_inverse', 'custom'],
                       default='balanced', help='Label weight calculation')
    parser.add_argument('--custom_label_weights', type=str, nargs='+', help='Custom weights')

    # CXR dropout
    parser.add_argument('--cxr_dropout_rate', type=float, default=0.0, help='CXR dropout during training')
    parser.add_argument('--cxr_dropout_seed', type=int, default=None, help='CXR dropout random seed')
    
    # SMIL-specific parameters
    parser.add_argument('--cxr_mean_path', type=str, default=None, 
                       help='Path to CXR k-means centers directory (for SMIL model). Default: models/smil/cxr_mean')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for CXR k-means (for SMIL model)')

    # Fairness-related
    parser.add_argument('--compute_fairness', action='store_true', help='Compute fairness metrics')
    parser.add_argument('--fairness_attributes', type=str, nargs='+',
                       default=["admission_type", "age", "gender", "race"],
                       help='Sensitive attributes for fairness')
    parser.add_argument('--fairness_age_bins', type=float, nargs='+',
                       default=[0, 40, 60, 80, float('inf')],
                       help='Age bins for fairness')
    parser.add_argument('--fairness_intersectional', action='store_true', help='Intersectional fairness')
    parser.add_argument('--fairness_include_cxr', action='store_true', help='Include CXR in fairness')

    # Prediction saving
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions')
    parser.add_argument('--predictions_save_dir', type=str, default=None, help='Predictions output directory')
    
    # Auto-find checkpoint
    parser.add_argument('--experiments_dir', type=str, default=None, 
                       help='Base directory for experiments (e.g., ./experiments or ./experiments-m-m). If set and checkpoint_path not provided, will automatically find best checkpoint based on model/task/fold/seed.')

    parser.add_argument('--task', type=str, default='mortality',help='phenotype or mortality')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')

    # Data paths
    parser.add_argument('--resized_cxr_root', type=str, default='/hdd/benchmark/benchmark_dataset/mimic_cxr_resized',
                        help='CXR data path')
    parser.add_argument('--image_meta_path', type=str, default='/hdd/benchmark/benchmark_dataset/mimic-cxr-2.0.0-metadata.csv',
                        help='CXR meta csv')
    parser.add_argument('--ehr_root', type=str, default='/hdd/benchmark/benchmark_dataset/DataProcessing/benchmark_data/250827',
                        help='EHR data path')
    parser.add_argument('--pkl_dir', type=str, default='/hdd/benchmark/benchmark_dataset/DataProcessing/benchmark_data/250827/data_pkls',
                        help='Data PKL dir')
    
    # parser.add_argument('--resized_cxr_root', type=str, help='Path to the cxr data',
    #                     default='/hdd/benchmark/benchmark_dataset/mimic_cxr_resized')
    
    # parser.add_argument('--image_meta_path', type=str, help='Path to the image meta data',
    #                     default='/hdd/benchmark/benchmark_dataset/mimic-cxr-2.0.0-metadata.csv')
    
    # parser.add_argument('--ehr_root', type=str, help='Path to the data dir',
    #                 default='/hdd/benchmark/benchmark_dataset/DataProcessing/benchmark_data/250827')

    # parser.add_argument('--pkl_dir', type=str, help='Path to the pkl data',
    #                     default='/hdd/benchmark/benchmark_dataset/DataProcessing/benchmark_data/250827/data_pkls')
    
    parser.add_argument('--demographics_in_model_input', action='store_true',
                        help='Use demographics in model input')

    # First parse to get model & unknown args
    partial_args, unknown_args = parser.parse_known_args()
    model_name = partial_args.model

    # Which config YAML
    if partial_args.config_path:
        config_path = partial_args.config_path
        print(f"Using custom config path: {config_path}")
    else:
        config_path = os.path.join(partial_args.config_root, f'{model_name}.yaml')
        print(f"Using default config path: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML config not found for model `{model_name}` at: {config_path}")

    yaml_config = load_yaml_config(config_path)
    print(f"Loading YAML config: {config_path}")

    # Handle CLI overrides of YAML-defined keys
    modified_params = {}
    if unknown_args:
        print(f"\nDetected command line argument overrides:")
        i = 0
        while i < len(unknown_args):
            if unknown_args[i].startswith('--'):
                param_name = unknown_args[i][2:]
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                    param_value = unknown_args[i + 1]
                    i += 2
                else:
                    # Boolean flag
                    param_value = 'True'
                    i += 1
                if param_name in yaml_config:
                    original_value = yaml_config[param_name]
                    # Type conversion
                    if isinstance(original_value, bool):
                        param_value = param_value.lower() in ('true', 'yes', 'y', '1')
                    elif isinstance(original_value, int):
                        param_value = int(param_value)
                    elif isinstance(original_value, float):
                        param_value = float(param_value)
                    yaml_config[param_name] = param_value
                    modified_params[param_name] = (original_value, param_value)
                    # Update parser
                    if isinstance(param_value, bool):
                        action = 'store_true' if param_value else 'store_false'
                        parser.add_argument(f'--{param_name}', action=action, default=param_value)
                    else:
                        parser.add_argument(f'--{param_name}', type=type(param_value), default=param_value)
                else:
                    print(f"  Warning: Parameter '{param_name}' not in YAML config, ignored")
            else:
                i += 1
                print(f"  Warning: Ignoring invalid parameter: {unknown_args[i-1]}")

    if modified_params:
        for param_name, (old_value, new_value) in modified_params.items():
            print(f"Override: {param_name} = {new_value} (original: {old_value})")
    else:
        print("No YAML parameters were modified")
    print("")

    parser.set_defaults(**yaml_config)
    args = parser.parse_args()
    
    # Track which arguments were explicitly provided in CLI
    # For action='store_true', if the flag is present, the value is True
    # We need to check sys.argv to see if the flag was actually provided
    import sys
    cli_provided_args = set()
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--'):
            arg_name = arg[2:]
            # Check if it's a store_true argument that was provided
            if arg_name in ['compute_fairness', 'fairness_intersectional', 'fairness_include_cxr', 
                           'save_predictions', 'use_demographics', 'use_triplet', 'matched', 
                           'dev_run', 'save_checkpoint', 'demographics_in_model_input', 'use_label_weights']:
                cli_provided_args.add(arg_name)
    
    # Fix for action='store_true' arguments: manually set them from YAML config
    # This is needed because action='store_true' ignores default values when the flag is not present in CLI
    # BUT: If CLI explicitly provided the flag, don't override it with YAML value
    store_true_args = ['compute_fairness', 'fairness_intersectional', 'fairness_include_cxr', 
                       'save_predictions', 'use_demographics', 'use_triplet', 'matched', 
                       'dev_run', 'save_checkpoint', 'demographics_in_model_input', 'use_label_weights']
    for arg_name in store_true_args:
        if arg_name in yaml_config and isinstance(yaml_config[arg_name], bool):
            # Only override if CLI didn't explicitly provide this argument
            if arg_name not in cli_provided_args:
                yaml_value = yaml_config[arg_name]
                current_value = getattr(args, arg_name, False)
                if yaml_value != current_value:
                    setattr(args, arg_name, yaml_value)
                    print(f"Set {arg_name} = {yaml_value} from YAML config (was {current_value})")
            else:
                # CLI provided this argument, keep CLI value
                cli_value = getattr(args, arg_name, False)
                yaml_value = yaml_config[arg_name]
                if cli_value != yaml_value:
                    print(f"Using CLI value {arg_name} = {cli_value} (YAML has {yaml_value}, but CLI takes precedence)")

    # Dataset split flags
    if args.cross_eval == 'matched_to_full':
        args.train_matched = True
        args.val_matched = True
        args.test_matched = False
        print("Cross evaluation: train matched, test full")
    elif args.cross_eval == 'full_to_matched':
        args.train_matched = False
        args.val_matched = False
        args.test_matched = True
        print("Cross evaluation: train full, test matched")
    elif args.matched:
        args.train_matched = True
        args.val_matched = True
        args.test_matched = True
        print("Using matched data for all splits")
    else:
        args.train_matched = False
        args.val_matched = False
        args.test_matched = False
        print("Using full data for all splits")

    # Demographics printout
    if args.use_demographics:
        print(f"Demographic features: {args.demographic_cols}")
    else:
        print("Demographic features disabled")

    return args