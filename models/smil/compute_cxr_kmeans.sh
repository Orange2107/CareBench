#!/bin/bash

# CXR K-means computation script for SMIL
# This script computes k-means cluster centers for CXR features across specified folds

# Set default parameters
TASK="phenotype"
FOLDS=(1 2 3 4 5)  # Default to all folds
CXR_ENCODER="resnet50"  
N_CLUSTERS=10
BATCH_SIZE=16
GPU=0
DATA_TYPE="full"  # matched or full
HF_MODEL_ID="codewithdark/vit-chest-xray"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --folds)
            # Read all subsequent arguments as fold numbers until next flag or end
            FOLDS=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FOLDS+=("$1")
                shift
            done
            ;;
        --cxr_encoder)
            CXR_ENCODER="$2"
            shift 2
            ;;
        --n_clusters)
            N_CLUSTERS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --data_type)
            DATA_TYPE="$2"
            shift 2
            ;;
        --hf_model_id)
            HF_MODEL_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --task          Task type (phenotype|mortality) [default: phenotype]"
            echo "  --folds         Fold numbers to process [default: 1 2 3 4 5]"
            echo "  --cxr_encoder   CXR encoder (resnet50|hf_chexpert_vit|densenet121-imagenet|densenet121-res224-chex) [default: resnet50]"
            echo "  --n_clusters    Number of clusters [default: 10]"
            echo "  --batch_size    Batch size [default: 16]"
            echo "  --gpu           GPU device ID [default: 0]"
            echo "  --data_type     Data type (matched|full) [default: full]"
            echo "  --hf_model_id   HF model id for hf_chexpert_vit [default: codewithdark/vit-chest-xray]"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Process all folds (1-5)"
            echo "  $0 --folds 1 3 5                          # Process folds 1, 3, 5"
            echo "  $0 --folds 2 4                            # Process folds 2, 4"
            echo "  $0 --folds 1 --task mortality             # Process fold 1 for mortality"
            echo "  $0 --folds 1 2 3 --data_type full         # Process folds 1-3 with full data"
            echo "  $0 --folds 1 --cxr_encoder hf_chexpert_vit --data_type full"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Computing CXR k-means cluster centers for SMIL"
echo "============================================"
echo "CXR Encoder: $CXR_ENCODER"
echo "Number of clusters: $N_CLUSTERS"
echo "Task: $TASK"
echo "Folds to process: ${FOLDS[*]}"
echo "Data type: $DATA_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "GPU: $GPU"
if [ "$CXR_ENCODER" = "hf_chexpert_vit" ]; then
    echo "HF model id: $HF_MODEL_ID"
fi
echo "============================================"

# Create output directory if it doesn't exist
mkdir -p ./cxr_mean

# Track success and failures
SUCCESSFUL_FOLDS=()
FAILED_FOLDS=()
START_TIME=$(date +%s)

# Loop through each specified fold
for FOLD in "${FOLDS[@]}"; do
    echo ""
    echo "Processing Fold $FOLD..."
    echo "----------------------------------------"
    
    # Validate fold number
    if [[ ! "$FOLD" =~ ^[1-5]$ ]]; then
        echo "✗ Invalid fold number: $FOLD (must be 1-5)"
        FAILED_FOLDS+=($FOLD)
        continue
    fi
    
    # Build the command
    CMD="python ./compute_cxr_mean_kmeans.py \
        --task $TASK \
        --fold $FOLD \
        --cxr_encoder $CXR_ENCODER \
        --pretrained \
        --hf_model_id $HF_MODEL_ID \
        --hidden_dim 256 \
        --n_clusters $N_CLUSTERS \
        --batch_size $BATCH_SIZE \
        --num_workers 4 \
        --use_minibatch \
        --gpu $GPU \
        --seed 42 \
        --output_dir ./cxr_mean \
        --output_name cxr_mean_fold${FOLD}_${DATA_TYPE}_${CXR_ENCODER}_${N_CLUSTERS}clusters.npy"

    # Add matched flag if data_type is matched
    if [ "$DATA_TYPE" = "matched" ]; then
        CMD="$CMD --matched"
    fi

    echo "Running command: $CMD"
    
    # Execute the command and capture the exit status
    if eval $CMD; then
        echo "✓ Fold $FOLD completed successfully!"
        SUCCESSFUL_FOLDS+=($FOLD)
    else
        echo "✗ Fold $FOLD failed!"
        FAILED_FOLDS+=($FOLD)
    fi
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "============================================"
echo "CXR k-means computation completed!"
echo "============================================"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Print successful folds
if [ ${#SUCCESSFUL_FOLDS[@]} -gt 0 ]; then
    echo "✓ Successfully processed folds: ${SUCCESSFUL_FOLDS[*]}"
    echo "Results saved to ./cxr_mean/ with naming pattern:"
    echo "  cxr_mean_fold{FOLD}_${DATA_TYPE}_${CXR_ENCODER}_${N_CLUSTERS}clusters.npy"
fi

# Print failed folds
if [ ${#FAILED_FOLDS[@]} -gt 0 ]; then
    echo "✗ Failed folds: ${FAILED_FOLDS[*]}"
    echo "Please check the logs above for error details."
    exit 1
fi

echo ""
echo "All specified folds processed successfully!"
