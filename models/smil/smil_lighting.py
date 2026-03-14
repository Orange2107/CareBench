import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import Dict, Any

from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from .smil_components import SMILEncoder, SMILLoss


@ModelRegistry.register('smil')
class SMIL(BaseFuseTrainer):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.task = self.hparams.task
        
        # Set task-specific number of classes
        if self.task == 'phenotype':
            self.num_classes = self.hparams.num_classes
        elif self.task == 'mortality':
            self.num_classes = 1
        elif self.task == 'los':
            self.num_classes = 7  # LoS has 7 classes (bins 2-8, excluding 0,1)
        else:
            raise ValueError(f"Unsupported task: {self.task}. Only 'mortality', 'phenotype', and 'los' are supported")
        
        # Initialize model components
        self.inner_loop = self.hparams.inner_loop
        self.lr_inner = self.hparams.lr_inner
        self.mc_size = self.hparams.mc_size
        
        # Initialize encoder 
        self.encoder = SMILEncoder(hparams)
        
        # Initialize loss functions
        self.criterion = SMILLoss(hparams)
        
        # Load pre-computed CXR mean (following the reference pattern)
        self._load_precomputed_cxr_mean()

    def _load_precomputed_cxr_mean(self):
        """Load pre-computed CXR k-means centers directly"""
        # Get base path - use absolute path from project root
        cxr_mean_path = getattr(self.hparams, 'cxr_mean_path', None)
        if cxr_mean_path is None:
            # Default to models/smil/cxr_mean relative to project root
            # Get project root (assuming we're in CareBench directory)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cxr_mean_path = os.path.join(project_root, 'models', 'smil', 'cxr_mean')
        
        # Generate CXR k-means file name dynamically based on configuration
        fold = getattr(self.hparams, 'fold', 1)
        cxr_encoder = getattr(self.hparams, 'cxr_encoder', 'resnet50')
        n_clusters = getattr(self.hparams, 'n_clusters', 10)
        
        # Determine data type based on matched parameters
        # Check train_matched first (set by arguments.py logic)
        if hasattr(self.hparams, 'train_matched'):
            data_type = 'matched' if self.hparams.train_matched else 'full'
        # Fallback to matched parameter
        elif hasattr(self.hparams, 'matched'):
            data_type = 'matched' if self.hparams.matched else 'full'
        else:
            # Default to matched for backward compatibility
            data_type = 'matched'
        
        # Generate file name: cxr_mean_fold{fold}_{data_type}_{encoder}_{clusters}clusters.npy
        cxr_mean_name = f"cxr_mean_fold{fold}_{data_type}_{cxr_encoder}_{n_clusters}clusters.npy"
        
        # Allow override if explicitly specified
        if hasattr(self.hparams, 'cxr_mean_name'):
            cxr_mean_name = self.hparams.cxr_mean_name
        
        full_path = os.path.join(cxr_mean_path, cxr_mean_name)
        # Ensure absolute path
        full_path = os.path.abspath(full_path)
        
        print(f"Data type determined: {data_type}")
        print(f"Expected CXR k-means file: {cxr_mean_name}")
        print(f"Searching for file at: {full_path}")
        
        # Try alternative paths if the default path doesn't exist
        if not os.path.exists(full_path):
            # Try alternative paths
            alternative_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'smil', 'cxr_mean', cxr_mean_name),
                os.path.join('./models/smil/cxr_mean', cxr_mean_name),
                os.path.join('../models/smil/cxr_mean', cxr_mean_name),
                os.path.join('models/smil/cxr_mean', cxr_mean_name),
            ]
            
            for alt_path in alternative_paths:
                alt_path_abs = os.path.abspath(alt_path)
                if os.path.exists(alt_path_abs):
                    print(f"Found file at alternative path: {alt_path_abs}")
                    print(f"Using this path instead of: {full_path}")
                    full_path = alt_path_abs
                    break
        
        if os.path.exists(full_path):
            print(f"Loading pre-computed CXR k-means centers from: {full_path}")
            try:
                # Load pre-computed k-means centers
                cxr_mean = np.load(full_path)
                # Convert to tensor and transpose (following reference pattern)
                cxr_mean_tensor = torch.from_numpy(cxr_mean).T.float()
                
                print(f"CXR k-means centers loaded successfully")
                print(f"  - Original shape: {cxr_mean.shape}")
                print(f"  - Transposed shape: {cxr_mean_tensor.shape}")
                print(f"  - Number of clusters: {cxr_mean.shape[0]}")
                print(f"  - Feature dimension: {cxr_mean.shape[1]}")
                
                # Register as buffer so it moves with the model to GPU
                self.register_buffer('cxr_mean', cxr_mean_tensor)
                
            except Exception as e:
                raise ValueError(f"Failed to load CXR k-means centers from {full_path}: {e}")
        else:
            # Provide clear guidance on how to generate the required file
            error_msg = f"""
            CXR k-means centers file not found: {full_path}
            
            Please generate the CXR k-means centers first using:
            
            1. Navigate to the SMIL directory:
               cd models/smil/
            
            2. Run the k-means computation script:
               python compute_cxr_mean_kmeans.py --task {getattr(self.hparams, 'task', 'phenotype')} --fold {fold} --data_type {data_type} --cxr_encoder {cxr_encoder} --n_clusters {n_clusters}
               
            Or use the shell script:
               ./compute_cxr_kmeans.sh --task {getattr(self.hparams, 'task', 'phenotype')} --folds {fold} --data_type {data_type} --cxr_encoder {cxr_encoder} --n_clusters {n_clusters}
            
            The script will generate the required file at: {full_path}
            
            Available parameters for debugging:
            - train_matched: {getattr(self.hparams, 'train_matched', 'NOT_SET')}
            - matched: {getattr(self.hparams, 'matched', 'NOT_SET')}
            - fold: {fold}
            - cxr_encoder: {cxr_encoder}
            - n_clusters: {n_clusters}
            """
            raise FileNotFoundError(error_msg)

    def _functional_forward(self, params, batch, mode='one'):
        """Functional forward pass for meta-learning"""
        # Create functional version of encoder
        encoder_func = self.encoder.functional(params, create_graph=True)
        
        # Forward pass
        if mode == 'one':
            # Reconstruct CXR features using EHR features
            pred, f, f1, cxr_features = encoder_func(batch, mode='one', 
                                                   cxr_mean=self.cxr_mean,
                                                   meta_train=batch.get('_meta_train', True))
        else:
            # Use complete modalities
            pred, f, f1, cxr_features = encoder_func(batch, mode='two',
                                                   meta_train=False)  # No noise during validation
        
        return pred, f, f1, cxr_features

    def _normalize_has_cxr(self, has_cxr, device):
        if has_cxr is None:
            return None
        if not isinstance(has_cxr, torch.Tensor):
            has_cxr = torch.as_tensor(has_cxr, device=device)
        return has_cxr.view(-1).bool().to(device)

    def _select_batch(self, batch, indices):
        subset = {}
        for key, value in batch.items():
            if key == '_meta_train':
                continue
            if isinstance(value, torch.Tensor):
                subset[key] = value[indices]
            else:
                subset[key] = value
        return subset

    def _build_missing_view(self, batch, meta_train):
        missing_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                missing_batch[key] = value.clone()
            else:
                missing_batch[key] = value

        if 'cxr_imgs' in missing_batch:
            missing_batch['cxr_imgs'] = torch.zeros_like(missing_batch['cxr_imgs'])
        if 'has_cxr' in missing_batch:
            missing_batch['has_cxr'] = torch.zeros_like(missing_batch['has_cxr'])
        missing_batch['_meta_train'] = meta_train
        return missing_batch

    def _select_complete_indices(self, candidate_indices, has_cxr):
        present_mask = has_cxr[candidate_indices]
        return candidate_indices[present_mask]

    def forward(self, batch):
        has_cxr = self._normalize_has_cxr(batch.get('has_cxr'), batch['ehr_ts'].device)
        if has_cxr is None or torch.all(has_cxr):
            pred, f, f1, cxr_features = self.encoder(batch, mode='two')
        else:
            pred, f, f1, cxr_features = self.encoder(
                batch,
                mode='one',
                cxr_mean=self.cxr_mean,
                meta_train=False
            )
        loss = self.criterion.meta_forward(pred, batch['labels'].squeeze())
        
        return {
            'loss': loss,
            'predictions': pred,
            'labels': batch['labels'].squeeze(),
            'feat_cxr_distinct': cxr_features,
            'feat_ehr_distinct': f
        }

    def training_step(self, batch, batch_idx):
        """Meta-learning training step"""
        # Get parameters
        params = list(self.parameters())
        params_star = params
        
        # Calculate split points
        batch_size = batch['cxr_imgs'].size(0)
        val_size = int(batch_size // 5)  # Use 20% of data as meta-val
        train_size = batch_size - val_size
        
        # Ensure seq_len is a tensor
        if not isinstance(batch['seq_len'], torch.Tensor):
            batch['seq_len'] = torch.tensor(batch['seq_len'], dtype=torch.int64)
        if not isinstance(batch['has_cxr'], torch.Tensor):
            batch['has_cxr'] = torch.tensor(batch['has_cxr'], dtype=torch.float32)

        has_cxr = self._normalize_has_cxr(batch['has_cxr'], batch['ehr_ts'].device)

        # Randomly split data
        indices = torch.randperm(batch_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        meta_train_base = self._select_batch(batch, train_indices)
        meta_train_split = self._build_missing_view(meta_train_base, meta_train=True)

        complete_val_indices = self._select_complete_indices(val_indices, has_cxr)
        if len(complete_val_indices) == 0:
            complete_val_indices = self._select_complete_indices(train_indices, has_cxr)
        if len(complete_val_indices) == 0:
            complete_val_indices = has_cxr.nonzero(as_tuple=True)[0]
        if len(complete_val_indices) == 0:
            raise ValueError("SMIL training requires at least one complete CXR sample in the batch for meta-val clean.")

        meta_val_clean_split = self._select_batch(batch, complete_val_indices)
        meta_val_clean_split['_meta_train'] = False
        meta_val_noised_split = self._build_missing_view(meta_val_clean_split, meta_train=False)
        
        # Meta-training inner loop
        loss_meta_train = 0.
        loss_meta_val = 0.
        mse_loss = nn.MSELoss(reduction='mean')
        
        for idx in range(self.inner_loop):
            if idx == 0:
                params_star = params
            
            # Forward pass (missing CXR modality)
            pred_meta_train_noised, f_meta_train_noised1, f_meta_train_noised2, cxr_mean_train = self._functional_forward(
                params_star,
                meta_train_split,
                mode='one'
            )
            
            # Calculate meta-train loss
            loss_meta_train = self.criterion.meta_forward(pred_meta_train_noised, meta_train_split['labels'].squeeze())
            
            # Calculate gradients
            grads = torch.autograd.grad(
                loss_meta_train, 
                params_star, 
                allow_unused=True, 
                create_graph=True
            )
            
            # Update parameters (excluding EHR branch)
            ehr_params_start = len(list(self.encoder.cxr_encoder.parameters()))
            ehr_params_end = ehr_params_start + len(list(self.encoder.ehr_encoder.parameters()))
            
            lr = self.lr_inner * (0.1 ** (self.current_epoch // 1000))
            
            for i in range(len(params_star)):
                if i < ehr_params_start or i >= ehr_params_end:  # Don't update EHR branch
                    if grads[i] is not None:
                        params_star[i] = (
                            params_star[i] - lr * grads[i]
                        ).requires_grad_()
        
        # Meta-validation with missing CXR
        pred_meta_val_noised, f_meta_val_noised1, f_meta_val_noised2, cxr_mean_val_noised = self._functional_forward(
            params_star,
            meta_val_noised_split,
            mode='one'
        )
        
        # Meta-validation with complete modalities
        pred_meta_val_clean, f_meta_val_clean1, f_meta_val_clean2, cxr_mean_val_clean = self._functional_forward(
            params_star,
            meta_val_clean_split,
            mode='two'
        )
        
        # Calculate MSE loss for CXR feature means
        cxr_mean_val_mse = mse_loss(cxr_mean_val_clean, cxr_mean_val_noised)
        
        # Calculate meta-validation loss
        loss_meta_val = self.criterion(
            cxr_mean_val_clean,
            cxr_mean_val_noised,
            f_meta_val_clean1,
            f_meta_val_noised1,
            f_meta_val_clean2,
            f_meta_val_noised2,
            pred_meta_val_noised,
            pred_meta_val_clean,
            meta_val_clean_split['labels'].squeeze()
        )
        
        # Total loss
        total_loss = loss_meta_train + loss_meta_val
        
        # Record metrics
        self.log('train/meta_train_loss', loss_meta_train, on_epoch=True, batch_size=train_size, sync_dist=True)
        self.log('train/meta_val_loss', loss_meta_val, on_epoch=True, batch_size=meta_val_clean_split['labels'].shape[0], sync_dist=True)
        self.log('train/cxr_mean_mse', cxr_mean_val_mse, on_epoch=True, batch_size=meta_val_clean_split['labels'].shape[0], sync_dist=True)
        
        return {
            "loss": total_loss,
            "pred": pred_meta_val_clean.detach(),
            "labels": meta_val_clean_split['labels'].squeeze().detach()
        }

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=getattr(self.hparams, 'lr', 0.0001)
        )

        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=getattr(self.hparams, 'patience', 10),
                mode='min',
                verbose=True
            ),
            "monitor": "loss/validation_epoch",
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
