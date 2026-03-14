"""
CareBench Main Entry Point
Main entry for training and testing multimodal clinical prediction models.
"""

import os
import pickle
import yaml
import random
import argparse
from copy import deepcopy
from datetime import datetime
from argparse import Namespace
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.init as init
from torch import nn
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from arguments import get_args
from models import get_model, get_model_cls
from utils.ver_name import get_version_name
from utils.checkpoint_finder import auto_find_checkpoint
from datasets.dataset import create_data_loaders


def get_device_map_location(target_device: int = 0) -> str:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if target_device >= device_count:
            target_device = 0
        return f'cuda:{target_device}'
    return 'cpu'


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(5)
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_checkpoint_hparams(model_from_checkpoint: Any, hparams: Dict, force_update_keys: list = None, protected_keys: list = None) -> None:
    """
    Update checkpoint hparams while protecting training parameters.
    
    Args:
        model_from_checkpoint: Model loaded from checkpoint
        hparams: Dictionary of hyperparameters to potentially update
        force_update_keys: List of keys that should be force-updated (test/evaluation related)
        protected_keys: List of keys that should NEVER be modified (training parameters)
    """
    if force_update_keys is None:
        force_update_keys = ['compute_fairness', 'fairness_attributes', 'fairness_age_bins', 
                            'fairness_intersectional', 'save_predictions', 'use_demographics',
                            'cxr_mean_path']
    
    if protected_keys is None:
        protected_keys = ['dropout', 'lr', 'batch_size', 'hidden_size', 'ehr_dropout', 
                         'ehr_n_head', 'ehr_n_layers', 'ehr_num_layers', 'ehr_bidirectional',
                         'pretrained', 'aug_merge_alpha', 'aug_lambda', 'aug_margin',
                         'aug_layer_check_interval', 'aug_classifier_hidden_dim',
                         'num_classes', 'input_dim', 'patience', 'epochs']
    
    for key, value in hparams.items():
        if key in force_update_keys:
            setattr(model_from_checkpoint.hparams, key, value)
            print(f"Force updated {key} = {value}")
        elif key in protected_keys:
            # Skip protected training parameters - keep checkpoint values
            checkpoint_value = getattr(model_from_checkpoint.hparams, key, None)
            if checkpoint_value is not None and checkpoint_value != value:
                print(f"Warning: Skipping update of protected parameter {key} (checkpoint: {checkpoint_value}, requested: {value})")
            continue
        elif not hasattr(model_from_checkpoint.hparams, key):
            setattr(model_from_checkpoint.hparams, key, value)
            print(f"Added new parameter {key} = {value}")


def sync_args_with_checkpoint(args: Namespace, model_from_checkpoint: Any) -> bool:
    """
    Sync args with checkpoint parameters to ensure data loading consistency.
    Returns True if data loaders need to be recreated.
    """
    needs_reload = False
    
    checkpoint_seed = getattr(model_from_checkpoint.hparams, 'seed', None)
    checkpoint_matched = getattr(model_from_checkpoint.hparams, 'matched', None)
    checkpoint_batch_size = getattr(model_from_checkpoint.hparams, 'batch_size', None)
    checkpoint_use_phenotype9 = getattr(model_from_checkpoint.hparams, 'use_phenotype9', None)
    checkpoint_num_classes = getattr(model_from_checkpoint.hparams, 'num_classes', None)
    
    # Update seed if different
    if checkpoint_seed is not None and checkpoint_seed != args.seed:
        print(f"Warning: Checkpoint seed ({checkpoint_seed}) differs from specified seed ({args.seed})")
        print(f"Using checkpoint seed ({checkpoint_seed}) for data loading to ensure reproducibility")
        args.seed = checkpoint_seed
        needs_reload = True
    
    # Update matched parameter if different - CRITICAL for data consistency
    # BUT: If cross_eval is set, allow test set to use different data (matched_to_full or full_to_matched)
    if checkpoint_matched is not None:
        current_matched = getattr(args, 'matched', True)
        cross_eval_mode = getattr(args, 'cross_eval', None)
        
        if checkpoint_matched != current_matched:
            print(f"Warning: Checkpoint matched={checkpoint_matched} differs from test matched={current_matched}")
            
            # If cross_eval is enabled, preserve the cross-evaluation test configuration
            if cross_eval_mode:
                print(f"Cross-evaluation mode ({cross_eval_mode}) is enabled, preserving test data configuration")
                # Only sync train and val, keep test_matched as set by cross_eval
                if checkpoint_matched:
                    args.train_matched = True
                    args.val_matched = True
                    # test_matched is already set by cross_eval, don't override it
                    print(f"Using matched data for train/val, {args.test_matched} for test (from cross_eval)")
                else:
                    args.train_matched = False
                    args.val_matched = False
                    # test_matched is already set by cross_eval, don't override it
                    print(f"Using full data for train/val, {args.test_matched} for test (from cross_eval)")
            else:
                # Normal mode: sync all splits with checkpoint
                print(f"Using checkpoint matched={checkpoint_matched} for data loading to ensure consistency")
                args.matched = checkpoint_matched
                # Update train_matched, val_matched, test_matched based on matched flag
                if checkpoint_matched:
                    args.train_matched = True
                    args.val_matched = True
                    args.test_matched = True
                    print("Updated to use matched data for all splits")
                else:
                    args.train_matched = False
                    args.val_matched = False
                    args.test_matched = False
                    print("Updated to use full data for all splits")
            needs_reload = True
    
    # Update batch_size if different - CRITICAL for data loading order consistency
    # Different batch_size can affect data loader order, especially with shuffling
    if checkpoint_batch_size is not None:
        current_batch_size = getattr(args, 'batch_size', None)
        if current_batch_size is not None and checkpoint_batch_size != current_batch_size:
            print(f"Warning: Checkpoint batch_size={checkpoint_batch_size} differs from test batch_size={current_batch_size}")
            print(f"Using checkpoint batch_size={checkpoint_batch_size} for data loading to ensure consistency")
            args.batch_size = checkpoint_batch_size
            needs_reload = True

    if checkpoint_use_phenotype9 is not None:
        current_use_phenotype9 = getattr(args, 'use_phenotype9', False)
        if checkpoint_use_phenotype9 != current_use_phenotype9:
            print(
                f"Warning: Checkpoint use_phenotype9={checkpoint_use_phenotype9} "
                f"differs from test use_phenotype9={current_use_phenotype9}"
            )
            print(f"Using checkpoint use_phenotype9={checkpoint_use_phenotype9} for data loading consistency")
            args.use_phenotype9 = checkpoint_use_phenotype9
            needs_reload = True

    if checkpoint_num_classes is not None:
        current_num_classes = getattr(args, 'num_classes', None)
        if current_num_classes != checkpoint_num_classes:
            print(
                f"Warning: Checkpoint num_classes={checkpoint_num_classes} "
                f"differs from requested num_classes={current_num_classes}"
            )
            print(f"Using checkpoint num_classes={checkpoint_num_classes}")
            args.num_classes = checkpoint_num_classes
            args.vision_num_classes = checkpoint_num_classes
    
    return needs_reload

def setup_data_loaders(args: Namespace) -> Tuple[Any, Any, Any]:
    print(f"Data subset configuration:")
    print(f"  Train: {'matched' if args.train_matched else 'full'} data")
    print(f"  Validation: {'matched' if args.val_matched else 'full'} data")
    print(f"  Test: {'matched' if args.test_matched else 'full'} data")

    # Encoder-specific image preprocessing (e.g., HF CheXpert ViT)
    use_chexpert_transform = False
    cxr_encoder = getattr(args, 'cxr_encoder', None)
    if isinstance(cxr_encoder, str) and cxr_encoder.lower() == 'hf_chexpert_vit':
        use_chexpert_transform = True
    vpt_feature = getattr(args, 'vpt_feature', None)
    if isinstance(vpt_feature, str) and (vpt_feature == 'hf_chexpert_vit' or vpt_feature.startswith('hf_')):
        use_chexpert_transform = True
    
    return create_data_loaders(
        args.ehr_root, args.task,
        args.fold, args.batch_size, args.num_workers,
        matched_subset=getattr(args, 'matched', False),
        train_matched=args.train_matched,
        val_matched=args.val_matched,
        test_matched=args.test_matched,
        use_triplet=args.use_triplet,
        seed=args.seed,
        resized_base_path=args.resized_cxr_root,
        image_meta_path=args.image_meta_path,
        pkl_dir=args.pkl_dir,
        use_demographics=args.use_demographics,
        demographic_cols=args.demographic_cols,
        use_label_weights=getattr(args, 'use_label_weights', True),
        label_weight_method=getattr(args, 'label_weight_method', 'balanced'),
        custom_label_weights=getattr(args, 'custom_label_weights', None),
        cxr_dropout_rate=getattr(args, 'cxr_dropout_rate', 0.0),
        cxr_dropout_seed=getattr(args, 'cxr_dropout_seed', None),
        demographics_in_model_input=getattr(args, 'demographics_in_model_input', False),
        use_chexpert_transform=use_chexpert_transform,
        use_phenotype9=getattr(args, 'use_phenotype9', False),
    )


def update_hparams_from_dataset(args: Namespace, hparams: Dict, train_loader: Any) -> None:
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'input_dim'):
        actual_input_dim = train_dataset.input_dim
        print(f"Auto-detected input dimension: {actual_input_dim}")
        args.input_dim = actual_input_dim
        hparams['input_dim'] = actual_input_dim
        
        if hasattr(train_dataset, 'base_ehr_dim') and hasattr(train_dataset, 'demo_feature_dim'):
            print(f"  Base EHR dimension: {train_dataset.base_ehr_dim}")
            print(f"  Demographic dimension: {train_dataset.demo_feature_dim}")
    else:
        print(f"Using configured input dimension: {args.input_dim}")
    
    hparams['class_names'] = train_dataset.CLASSES
    actual_num_classes = len(train_dataset.CLASSES)
    args.num_classes = actual_num_classes
    hparams['num_classes'] = actual_num_classes
    print(f"Auto-detected number of classes: {actual_num_classes}")
    args.vision_num_classes = actual_num_classes
    hparams['vision_num_classes'] = actual_num_classes
    print(f"Synced vision_num_classes to: {actual_num_classes}")
    hparams['steps_per_epoch'] = len(train_loader)
    
    if getattr(args, 'use_label_weights', False) and hasattr(train_dataset, 'get_label_weights'):
        label_weights = train_dataset.get_label_weights()
        if label_weights is not None:
            if isinstance(label_weights, torch.Tensor):
                hparams['label_weights'] = label_weights.detach().cpu().numpy().tolist()
            else:
                hparams['label_weights'] = label_weights
            print(f"Label weights loaded from dataset: {hparams['label_weights']}")
            if args.task == 'mortality':
                if isinstance(label_weights, torch.Tensor):
                    pos_weight = label_weights[1].item() if len(label_weights) > 1 else label_weights[0].item()
                else:
                    pos_weight = label_weights[1] if len(label_weights) > 1 else label_weights[0]
                hparams['mortality_pos_weight'] = pos_weight
                print(f"Mortality pos_weight: {pos_weight}")


def get_callback_config(task: str) -> Tuple[str, str]:
    if task == 'mortality':
        return 'overall/PRAUC', '{epoch:02d}-{overall/PRAUC:.2f}'
    elif task == 'phenotype':
        return 'overall/PRAUC', '{epoch:02d}-{overall/PRAUC:.2f}'
    elif task == 'los':
        return 'overall/ACC', '{epoch:02d}-{overall/ACC:.2f}'
    else:
        raise ValueError(f"Unknown task: {task}")


def run_model(args: Namespace) -> None:
    if isinstance(args, dict):
        args = Namespace(**args)
    hparams = vars(args)
    
    if args.mode == 'test':
        train_loader, val_loader, test_loader = None, None, None
        model = None  # Will be loaded from checkpoint
    else:
        set_seed(args.seed)
        train_loader, val_loader, test_loader = setup_data_loaders(args)
        update_hparams_from_dataset(args, hparams, train_loader)
        model = get_model(args.model, hparams)
    callback_metric, filename_template = get_callback_config(args.task)
    early_stop_callback = EarlyStopping(monitor=callback_metric,
                                    min_delta=0.00,
                                    patience=args.patience,
                                    verbose=False,
                                    mode="max")
    checkpoint_callback = ModelCheckpoint(
        monitor=callback_metric,
        mode='max',
        save_top_k=1,
        verbose=True,
        filename=filename_template
    )
    log_dir, ver_name = get_version_name(args)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"in the ver_name {ver_name}")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, version=ver_name)
    if args.mode == 'test':
        csv_logger = pl_loggers.CSVLogger(save_dir=log_dir, version=ver_name)
        loggers = [tb_logger, csv_logger]
    else:
        csv_logger = None
        loggers = [tb_logger]

    if len(args.gpu) > 1:
        devices = args.gpu
        strategy = 'ddp_find_unused_parameters_true'  
    else:
        devices = [args.gpu[0]]  
        strategy = 'auto'
        
    target_device = args.gpu[0] if isinstance(args.gpu, list) else args.gpu
    map_location = get_device_map_location(target_device)
        
    trainer = L.Trainer(
        enable_checkpointing=args.save_checkpoint,
        accelerator='gpu',
        devices=devices,
        strategy=strategy,
        fast_dev_run=20 if args.dev_run else False,
        logger=loggers,
        num_sanity_val_steps=0,
        max_epochs=args.epochs,
        log_every_n_steps=1,
        min_epochs=4,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    if args.mode == 'train':
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print("Test model")
        best_model_path = checkpoint_callback.best_model_path
        print(f"best_model_path: {best_model_path}")

        if not args.dev_run:
            if trainer.global_rank == 0:
                test_trainer = L.Trainer(
                    accelerator='gpu',
                    devices=[devices[0] if isinstance(devices, list) else devices],
                    logger=loggers,
                    enable_checkpointing=False,
                )
                best_model_path = checkpoint_callback.best_model_path
                print(f"ModelCheckpoint best_model_path: {best_model_path}")
                manual_best_path = getattr(model, 'best_model_path', None)
                print(f"Manual best_model_path: {manual_best_path}")
                model_to_test = None
                if manual_best_path and os.path.exists(manual_best_path):
                    print(f"Loading manually saved best model: {manual_best_path}")
                    model_to_test = get_model_cls(args.model).load_from_checkpoint(
                        manual_best_path, map_location=map_location
                    )
                    for key, value in hparams.items():
                        if not hasattr(model_to_test.hparams, key):
                            setattr(model_to_test.hparams, key, value)
                    print(f"Updated model hparams with save_predictions={getattr(model_to_test.hparams, 'save_predictions', False)}")
                elif best_model_path and os.path.exists(best_model_path):
                    print(f"Loading ModelCheckpoint best model: {best_model_path}")
                    model_to_test = get_model_cls(args.model).load_from_checkpoint(
                        best_model_path, map_location=map_location
                    )
                    for key, value in hparams.items():
                        if not hasattr(model_to_test.hparams, key):
                            setattr(model_to_test.hparams, key, value)
                    print(f"Updated model hparams with save_predictions={getattr(model_to_test.hparams, 'save_predictions', False)}")
                else:
                    print(f"Using current trained model for testing")
                    model_to_test = model
                
                test_trainer.test(model=model_to_test, dataloaders=test_loader)
                test_results = model_to_test.test_results
                # Use tb_logger to get log_dir if csv_logger is None (training mode)
                logger_for_save = csv_logger if csv_logger is not None else tb_logger
                save_test_results(logger_for_save, test_results)

    elif args.mode == 'test':
        print("Test mode")
        if not args.dev_run:
            if trainer.global_rank == 0:
                test_trainer = L.Trainer(
                    accelerator='gpu',
                    devices=[devices[0] if isinstance(devices, list) else devices],
                    logger=loggers,
                    enable_checkpointing=False,
                )
                model_to_test = None
                checkpoint_path_to_use = None
                
                # Auto-find checkpoint if experiments_dir is set and no checkpoint_path provided
                if (hasattr(args, 'experiments_dir') and args.experiments_dir and 
                    (not hasattr(args, 'checkpoint_path') or not args.checkpoint_path)):
                    experiments_base = getattr(args, 'experiments_dir', './experiments')
                    print(f"Auto-finding checkpoint from experiments directory: {experiments_base}")
                    
                    # Strategy: First extract actual parameters from directory name (more reliable than config)
                    # Then use these parameters to find the checkpoint
                    import glob
                    import re
                    
                    # Try to find any directory with matching seed
                    search_pattern = os.path.join(experiments_base, args.model, args.task, "lightning_logs", f"*seed_{args.seed}*")
                    potential_dirs = glob.glob(search_pattern)
                    
                    extracted_params = {}
                    
                    if potential_dirs:
                        # Use the first matching directory to extract parameters
                        dir_name = os.path.basename(potential_dirs[0])
                        print(f"Found potential directory: {dir_name[:100]}...")
                        
                        # Extract parameters from directory name (these are the actual training parameters)
                        # Extract batch_size
                        match = re.search(r'batch_size_(\d+)', dir_name)
                        if match:
                            extracted_params['batch_size'] = int(match.group(1))
                        
                        # Extract lr
                        match = re.search(r'lr_([\d.]+)', dir_name)
                        if match:
                            extracted_params['lr'] = float(match.group(1))
                        
                        # Extract dropout (if present)
                        match = re.search(r'dropout_([\d.]+)', dir_name)
                        if match:
                            extracted_params['dropout'] = float(match.group(1))
                        
                        # Extract pretrained (if present)
                        if 'pretrained_True' in dir_name:
                            extracted_params['pretrained'] = True
                        elif 'pretrained_False' in dir_name:
                            extracted_params['pretrained'] = False
                    
                    # Build search kwargs: prefer extracted params, fallback to config/defaults
                    search_kwargs = {
                        'base_dir': experiments_base,
                        'model': args.model,
                        'task': args.task,
                        'fold': args.fold,
                        'seed': args.seed,
                        'matched': getattr(args, 'matched', True),
                        'use_demographics': getattr(args, 'use_demographics', False),
                        'pretrained': extracted_params.get('pretrained', getattr(args, 'pretrained', True)),
                        'batch_size': extracted_params.get('batch_size', getattr(args, 'batch_size', 16)),
                        'lr': extracted_params.get('lr', getattr(args, 'lr', 0.0001)),
                        'patience': getattr(args, 'patience', 10),
                        'epochs': getattr(args, 'epochs', 50),
                    }
                    
                    # Only include dropout if it was extracted (some models don't have it in directory name)
                    if 'dropout' in extracted_params:
                        search_kwargs['dropout'] = extracted_params['dropout']
                    elif getattr(args, 'dropout', None) is not None:
                        # Only use config dropout if directory doesn't have it
                        search_kwargs['dropout'] = getattr(args, 'dropout', None)
                    
                    if extracted_params:
                        print(f"Using parameters extracted from directory name: {extracted_params}")
                    
                    checkpoint_path_to_use = auto_find_checkpoint(**search_kwargs)
                    
                    if checkpoint_path_to_use:
                        print(f"Auto-found checkpoint: {checkpoint_path_to_use}")
                    else:
                        print(f"Warning: Could not auto-find checkpoint, will try other methods")
                
                # Use manually specified checkpoint if provided
                if hasattr(args, 'checkpoint_path') and args.checkpoint_path and os.path.exists(args.checkpoint_path):
                    checkpoint_path_to_use = args.checkpoint_path
                    print(f"Using manually specified checkpoint: {checkpoint_path_to_use}")
                
                if checkpoint_path_to_use and os.path.exists(checkpoint_path_to_use):
                    print(f"Loading checkpoint: {checkpoint_path_to_use}")
                    model_to_test = get_model_cls(args.model).load_from_checkpoint(
                        checkpoint_path_to_use, map_location=map_location
                    )
                    
                    # Print checkpoint info for debugging
                    checkpoint_seed = getattr(model_to_test.hparams, 'seed', None)
                    checkpoint_matched = getattr(model_to_test.hparams, 'matched', None)
                    checkpoint_task = getattr(model_to_test.hparams, 'task', None)
                    checkpoint_fold = getattr(model_to_test.hparams, 'fold', None)
                    print(f"Checkpoint info: seed={checkpoint_seed}, task={checkpoint_task}, fold={checkpoint_fold}, matched={checkpoint_matched}")
                    print(f"Test args: seed={args.seed}, task={args.task}, fold={args.fold}, matched={getattr(args, 'matched', None)}")
                    
                    # Sync args with checkpoint parameters for data loading consistency
                    needs_data_reload = sync_args_with_checkpoint(args, model_to_test)
                    
                    # Create or recreate data loaders with correct parameters
                    if needs_data_reload or train_loader is None:
                        set_seed(args.seed)
                        train_loader, val_loader, test_loader = setup_data_loaders(args)
                        print(f"Created data loaders with seed={args.seed}, matched={getattr(args, 'matched', None)}")
                    else:
                        print(f"Using existing data loaders (parameters match checkpoint)")
                    
                    # Update hparams while protecting training parameters
                    update_checkpoint_hparams(model_to_test, hparams)
                    print(f"Updated model hparams with save_predictions={getattr(model_to_test.hparams, 'save_predictions', False)}")
                else:
                    best_model_path = checkpoint_callback.best_model_path
                    print(f"ModelCheckpoint best_model_path: {best_model_path}")
                    manual_best_path = getattr(model, 'best_model_path', None)
                    print(f"Manual best_model_path: {manual_best_path}")
                    if manual_best_path and os.path.exists(manual_best_path):
                        print(f"Loading manually saved best model: {manual_best_path}")
                        model_to_test = get_model_cls(args.model).load_from_checkpoint(
                            manual_best_path, map_location=map_location
                        )
                        # Sync args with checkpoint parameters
                        needs_data_reload = sync_args_with_checkpoint(args, model_to_test)
                        if needs_data_reload or train_loader is None:
                            set_seed(args.seed)
                            train_loader, val_loader, test_loader = setup_data_loaders(args)
                            print(f"Created data loaders with seed={args.seed}, matched={getattr(args, 'matched', None)}")
                        # Update hparams while protecting training parameters
                        update_checkpoint_hparams(model_to_test, hparams)
                        print(f"Updated model hparams with save_predictions={getattr(model_to_test.hparams, 'save_predictions', False)}")
                    elif best_model_path and os.path.exists(best_model_path):
                        print(f"Loading ModelCheckpoint best model: {best_model_path}")
                        model_to_test = get_model_cls(args.model).load_from_checkpoint(
                            best_model_path, map_location=map_location
                        )
                        # Sync args with checkpoint parameters
                        needs_data_reload = sync_args_with_checkpoint(args, model_to_test)
                        if needs_data_reload or train_loader is None:
                            set_seed(args.seed)
                            train_loader, val_loader, test_loader = setup_data_loaders(args)
                            print(f"Created data loaders with seed={args.seed}, matched={getattr(args, 'matched', None)}")
                        # Update hparams while protecting training parameters
                        
                        update_checkpoint_hparams(model_to_test, hparams)
                        print(f"Updated model hparams with save_predictions={getattr(model_to_test.hparams, 'save_predictions', False)}")
                    else:
                        # In test mode, model is None (not created yet)
                        # If no checkpoint found, we cannot proceed
                        raise ValueError(
                            f"Cannot test model: No checkpoint found and no trained model available.\n"
                            f"  - Checkpoint path: {checkpoint_path_to_use if checkpoint_path_to_use else 'Not specified'}\n"
                            f"  - Experiments dir: {getattr(args, 'experiments_dir', 'Not specified')}\n"
                            f"  - Model: {args.model}, Task: {args.task}, Fold: {args.fold}, Seed: {args.seed}\n"
                            f"  - Matched: {getattr(args, 'matched', None)}\n"
                            f"  - Training params: batch_size={getattr(args, 'batch_size', None)}, "
                            f"lr={getattr(args, 'lr', None)}, patience={getattr(args, 'patience', None)}, "
                            f"epochs={getattr(args, 'epochs', None)}, dropout={getattr(args, 'dropout', None)}\n"
                            f"\nPlease ensure:\n"
                            f"  1. Checkpoint exists at the specified path, OR\n"
                            f"  2. Use --experiments_dir with correct parameters matching the training config, OR\n"
                            f"  3. Specify --checkpoint_path directly"
                        )
                test_trainer.test(model=model_to_test, dataloaders=test_loader)
                test_results = model_to_test.test_results
                save_test_results(csv_logger, test_results)


def save_test_results(logger, test_results: Any) -> None:
    # Get log_dir from logger (can be CSVLogger or TensorBoardLogger)
    if logger is not None and hasattr(logger, 'log_dir'):
        save_path = os.path.join(logger.log_dir, 'test_set_results.yaml')
    else:
        # Fallback: use default path
        save_path = 'test_set_results.yaml'
    with open(save_path, 'w') as f:
        yaml.dump(test_results, f)
    print(f"Results saved to {save_path}")
    print(test_results)
    print("Save success!")


def run_experiments_from_config(config_path: str, mode: str = 'train', additional_args: Optional[list] = None) -> None:
    import yaml
    import sys
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading config: {config_path}")
    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"Description: {config.get('description', 'No description')}")
    config_mode = config.get('mode', mode)
    print(f"Mode: {config_mode}")
    
    cli_overrides = {}
    if additional_args:
        print(f"Processing CLI overrides: {additional_args}")
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--compute_fairness', action='store_true')
        parser.add_argument('--use_demographics', action='store_true')
        parser.add_argument('--save_predictions', action='store_true')
        parser.add_argument('--fairness_intersectional', action='store_true')
        parser.add_argument('--use_phenotype9', action='store_true')
        parser.add_argument('--patience', type=int)
        parser.add_argument('--dropout', type=float)

        # Encoder overrides (for switching CXR encoders globally)
        parser.add_argument('--cxr_encoder', type=str)
        parser.add_argument('--hf_model_id', type=str)
        parser.add_argument('--freeze_vit', type=str)
        parser.add_argument('--bias_tune', type=str)
        parser.add_argument('--partial_layers', type=int)
        
        try:
            parsed_overrides, _ = parser.parse_known_args(additional_args)
            cli_overrides = {k: v for k, v in vars(parsed_overrides).items() if v is not None}
            if cli_overrides:
                print(f"CLI overrides applied: {cli_overrides}")
        except Exception as e:
            print(f"Warning: Could not parse CLI overrides: {e}")
    
    def run_single_experiment(task, fold, seed, extra_config=None):
        original_argv = sys.argv.copy()
        try:
            sys.argv = ['main.py', '--model', config.get('model', 'drfuse'), 
                        '--config_path', config_path]
            args = get_args()
            meta_keys = {'tasks', 'folds', 'seeds', 'experiment_name', 'description', 'checkpoint_paths'}
            for key, value in config.items():
                if key not in meta_keys:
                    setattr(args, key, value)
            if extra_config:
                for key, value in extra_config.items():
                    setattr(args, key, value)
            for key, value in cli_overrides.items():
                setattr(args, key, value)
                print(f"  CLI override: {key} = {value}")
            args.task = task
            args.fold = fold
            args.seed = seed
            args.mode = config_mode
            
            # Handle checkpoint paths
            # Option 1: Per-seed checkpoint paths
            if 'checkpoint_paths' in config and isinstance(config['checkpoint_paths'], dict):
                if seed in config['checkpoint_paths']:
                    args.checkpoint_path = config['checkpoint_paths'][seed]
                    print(f"  Using seed-specific checkpoint: {args.checkpoint_path}")
            # Option 2: Single checkpoint path for all seeds
            elif 'checkpoint_path' in config and config.get('checkpoint_path'):
                args.checkpoint_path = config['checkpoint_path']
                print(f"  Using checkpoint: {args.checkpoint_path}")
            # Option 3: Auto-find checkpoint from experiments directory
            elif 'experiments_dir' in config and config.get('experiments_dir'):
                args.experiments_dir = config['experiments_dir']
                print(f"  Will auto-find checkpoint from: {args.experiments_dir}")
            
            run_model(args)
        finally:
            sys.argv = original_argv

    if 'tasks' in config and isinstance(config['tasks'], dict):
        tasks = config['tasks']
        folds = config.get('folds', [1])
        seeds = config.get('seeds', [42])
        total_experiments = len(tasks) * len(folds) * len(seeds)
        print(f"Running {total_experiments} {config_mode} experiments:")
        print(f"  Tasks: {list(tasks.keys())}")
        print(f"  Folds: {folds}")
        print(f"  Seeds: {seeds}\n")
        experiment_count = 0
        for task_name, task_config in tasks.items():
            for fold in folds:
                for seed in seeds:
                    experiment_count += 1
                    print(f"{'='*60}")
                    print(f"{config_mode.title()} Experiment {experiment_count}/{total_experiments}: {task_name} fold {fold} seed {seed}")
                    print(f"{'='*60}")
                    try:
                        run_single_experiment(task_name, fold, seed, task_config)
                        print(f"✓ Experiment {experiment_count} completed!\n")
                    except Exception as e:
                        print(f"✗ Experiment {experiment_count} failed: {e}\n")
                        import traceback
                        traceback.print_exc()
                        continue
        print(f"{'='*60}")
        print(f"All {total_experiments} {config_mode} experiments completed!")
        print(f"{'='*60}")
    
    elif 'seeds' in config and isinstance(config['seeds'], list):
        seeds = config['seeds']
        task = config.get('task', 'phenotype')
        fold = config.get('fold', 1)
        print(f"Running {len(seeds)} {config_mode} experiments with seeds: {seeds}\n")
        for i, seed in enumerate(seeds, 1):
            print(f"{'='*60}")
            print(f"{config_mode.title()} Experiment {i}/{len(seeds)}: seed {seed}")
            print(f"{'='*60}")
            try:
                run_single_experiment(task, fold, seed)
                print(f"✓ Experiment {i} completed!\n")
            except Exception as e:
                print(f"✗ Experiment {i} failed: {e}\n")
                import traceback
                traceback.print_exc()
                continue
    else:
        task = config.get('task', 'phenotype')
        fold = config.get('fold', 1)
        seed = config.get('seed', 42)
        print(f"Running single {config_mode} experiment\n")
        print(f"{'='*60}")
        print(f"{config_mode.title()} Experiment: {task} fold {fold} seed {seed}")
        print(f"{'='*60}")
        run_single_experiment(task, fold, seed)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--train_config':
        if len(sys.argv) < 3:
            print(f"Error: --train_config requires a config file path")
            print(f"Usage: python main.py --train_config <config_file.yaml> [additional args...]")
            print(f"Example: python main.py --train_config config.yaml --lr 0.001 --batch_size 32")
            sys.exit(1)
        config_path = sys.argv[2]
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        additional_args = sys.argv[3:] if len(sys.argv) > 3 else None
        if additional_args:
            print(f"Additional arguments detected: {additional_args}")
        run_experiments_from_config(config_path, mode='train', additional_args=additional_args)
    else:
        args = get_args()
        # Handle multiple seeds
        if isinstance(args.seed, list) and len(args.seed) > 1:
            seeds = args.seed
            print(f"Running {len(seeds)} experiments with seeds: {seeds}\n")
            for i, seed in enumerate(seeds, 1):
                print(f"{'='*60}")
                print(f"Experiment {i}/{len(seeds)}: seed {seed}")
                print(f"{'='*60}")
                # Create a copy of args with single seed
                import copy
                args_single = copy.deepcopy(args)
                args_single.seed = seed
                try:
                    run_model(args_single)
                    print(f"✓ Experiment {i} completed!\n")
                except Exception as e:
                    print(f"✗ Experiment {i} failed: {e}\n")
                    import traceback
                    traceback.print_exc()
                    continue
            print(f"{'='*60}")
            print(f"All {len(seeds)} experiments completed!")
            print(f"{'='*60}")
        else:
            # Single seed (convert list to int if needed)
            if isinstance(args.seed, list):
                args.seed = args.seed[0]
            run_model(args)
