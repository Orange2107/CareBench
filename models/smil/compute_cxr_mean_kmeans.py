from __future__ import print_function, absolute_import, division

import os
import time
import math
import datetime
import argparse
import os.path as path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans

# Add project root to Python path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import the project's data loader and encoder factory
from datasets.dataset import create_data_loaders
from models.base.base_encoder import create_cxr_encoder

def parse_args():
    parser = argparse.ArgumentParser(description='Compute CXR k-means for SMIL')
    
    # Data paths
    parser.add_argument('--ehr_root', type=str, 
                       default='/home/shared/benchmark_dataset/DataProcessing/benchmark_data/250430',
                       help='Path to the EHR data dir')
    parser.add_argument('--resized_cxr_root', type=str, 
                       default='/research/mimic_cxr_resized',
                       help='Path to the CXR data')
    parser.add_argument('--pkl_dir', type=str, 
                       default='/home/shared/benchmark_dataset/DataProcessing/benchmark_data/250430/data_pkls',
                       help='Path to the pkl data')
    parser.add_argument('--image_meta_path', type=str, 
                       default='/hdd/datasets/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv',
                       help='Path to the image meta data')
    
    # Output paths
    parser.add_argument('--output_dir', type=str, 
                       default='./cxr_mean',
                       help='Directory to save CXR mean')
    parser.add_argument('--output_name', type=str, 
                       default='cxr_mean.npy',
                       help='Filename for CXR mean')
    
    # Model parameters
    parser.add_argument('--cxr_encoder', type=str, default='resnet50',
                       choices=['resnet50'],
                       help='CXR encoder type')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for encoder')
    
    # Training parameters
    parser.add_argument('--task', type=str, default='phenotype',
                       choices=['phenotype', 'mortality'],
                       help='Task type')
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help='Data fold')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--matched', action='store_true',
                       help='Use matched subset (if not set, uses full dataset)')
    
    # K-means parameters
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters for k-means')
    parser.add_argument('--use_minibatch', action='store_true',
                       help='Use MiniBatchKMeans for large datasets')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for k-means')
    
    # Device settings
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()
    return args

def setup_device_and_seed(args):
    """Setup device and random seed"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    cudnn.benchmark = True
    return device

def create_cxr_feature_extractor(args, device):
    """Create CXR feature extractor using the factory function"""
    print(f'==> Creating CXR encoder: {args.cxr_encoder}')
    
    # Create CXR encoder using factory
    cxr_encoder = create_cxr_encoder(
        encoder_type=args.cxr_encoder,
        hidden_size=args.hidden_dim,
        pretrained=args.pretrained
    )
    
    cxr_encoder = cxr_encoder.to(device)
    cxr_encoder.eval()  # Set to evaluation mode
    
    print(f'CXR encoder created: {args.cxr_encoder}')
    print(f'  - Pretrained: {args.pretrained}')
    print(f'  - Hidden dimension: {args.hidden_dim}')
    
    return cxr_encoder

def extract_cxr_features(data_loader, cxr_encoder, device, args):
    """Extract CXR features from the dataset"""
    data_type = "matched" if args.matched else "full"
    print(f'==> Extracting CXR features from {data_type} dataset...')
    
    all_features = []
    total_samples = 0
    valid_samples = 0
    
    # Use tqdm for progress bar
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting features")):
            # Get CXR images and availability mask
            cxr_imgs = batch['cxr_imgs'].to(device)
            has_cxr = batch['has_cxr']  # Boolean mask for CXR availability
            
            # Only process samples that have CXR images
            valid_indices = has_cxr.nonzero(as_tuple=True)[0]
            
            if len(valid_indices) > 0:
                valid_cxr_imgs = cxr_imgs[valid_indices]
                
                # Extract features using the CXR encoder
                features = cxr_encoder(valid_cxr_imgs)
                
                # Flatten features if necessary
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                
                all_features.append(features.cpu())
                valid_samples += len(valid_indices)
            
            total_samples += len(cxr_imgs)
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f'Processed {batch_idx + 1} batches, '
                      f'Valid CXR samples: {valid_samples}/{total_samples}')
    
    if len(all_features) == 0:
        raise ValueError("No valid CXR features extracted!")
    
    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)
    print(f'Total CXR features extracted: {all_features.shape}')
    print(f'Valid CXR samples: {valid_samples}/{total_samples} '
          f'({100.0 * valid_samples / total_samples:.2f}%)')
    
    return all_features.numpy()

def perform_kmeans_clustering(features, args):
    """Perform k-means clustering on the extracted features"""
    print(f'==> Performing k-means clustering with {args.n_clusters} clusters...')
    
    if args.use_minibatch:
        print('Using MiniBatchKMeans for large dataset')
        kmeans = MiniBatchKMeans(
            n_clusters=args.n_clusters, 
            random_state=args.random_state,
            batch_size=1000,  # Batch size for MiniBatchKMeans
            max_iter=100
        )
    else:
        print('Using standard KMeans')
        kmeans = KMeans(
            n_clusters=args.n_clusters, 
            random_state=args.random_state,
            max_iter=300
        )
    
    # Fit k-means
    print('Fitting k-means...')
    kmeans.fit(features)
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    print(f'K-means clustering completed')
    print(f'Cluster centers shape: {cluster_centers.shape}')
    print(f'Inertia: {kmeans.inertia_:.2f}')
    
    return cluster_centers

def save_cluster_centers(cluster_centers, args):
    """Save cluster centers to file"""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save cluster centers
    save_path = os.path.join(args.output_dir, args.output_name)
    np.save(save_path, cluster_centers)
    
    print(f'==> Cluster centers saved to: {save_path}')
    print(f'Cluster centers shape: {cluster_centers.shape}')
    
    # Also save some metadata
    metadata_name = args.output_name.replace('.npy', '_metadata.txt')
    metadata_path = os.path.join(args.output_dir, metadata_name)
    
    data_type = "matched" if args.matched else "full"
    
    with open(metadata_path, 'w') as f:
        f.write(f'CXR K-means Clustering Metadata\n')
        f.write(f'=' * 40 + '\n')
        f.write(f'Generated at: {datetime.datetime.now()}\n')
        f.write(f'CXR encoder: {args.cxr_encoder}\n')
        f.write(f'Pretrained: {args.pretrained}\n')
        f.write(f'Hidden dimension: {args.hidden_dim}\n')
        f.write(f'Task: {args.task}\n')
        f.write(f'Fold: {args.fold}\n')
        f.write(f'Data type: {data_type}\n')
        f.write(f'Number of clusters: {args.n_clusters}\n')
        f.write(f'Use MiniBatch: {args.use_minibatch}\n')
        f.write(f'Random state: {args.random_state}\n')
        f.write(f'Cluster centers shape: {cluster_centers.shape}\n')
    
    print(f'Metadata saved to: {metadata_path}')

def main(args):
    data_type = "matched" if args.matched else "full"
    print('==> Starting CXR k-means computation for SMIL')
    print(f'Project root: {project_root}')
    print(f'Data type: {data_type}')
    print(f'Arguments: {args}')
    
    # Setup device and seed
    device = setup_device_and_seed(args)
    print(f'Using device: {device}')
    
    # Create data loaders
    print(f'==> Creating data loaders for {data_type} dataset...')
    train_loader, _, _ = create_data_loaders(
        args.ehr_root, args.task,
        args.fold, args.batch_size, args.num_workers,
        matched_subset=args.matched,
        train_matched=args.matched,
        val_matched=args.matched,
        test_matched=args.matched,
        use_triplet=False,
        seed=args.seed,
        resized_base_path=args.resized_cxr_root,
        image_meta_path=args.image_meta_path,
        pkl_dir=args.pkl_dir,
        use_demographics=False,
        demographic_cols=[]
    )
    
    print(f'Training data loader created with {len(train_loader)} batches')
    
    # Create CXR feature extractor
    cxr_encoder = create_cxr_feature_extractor(args, device)
    
    # Extract CXR features
    features = extract_cxr_features(train_loader, cxr_encoder, device, args)
    
    # Perform k-means clustering
    cluster_centers = perform_kmeans_clustering(features, args)
    
    # Save results
    save_cluster_centers(cluster_centers, args)
    
    print(f'==> CXR k-means computation completed successfully for {data_type} dataset!')

if __name__ == '__main__':
    args = parse_args()
    main(args) 