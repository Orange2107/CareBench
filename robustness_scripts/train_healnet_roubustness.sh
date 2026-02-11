#!/bin/bash

# =====================================================================
# HealNet CXR Dropout Robustness Experiment
# =====================================================================
# 
# Test HealNet model performance with different CXR dropout rates
# Uses the same parameters as the original HealNet training script
# =====================================================================

# =====================================================================
# Basic Configuration
# =====================================================================

# Basic training parameters
BATCH_SIZE=4
LR=0.0001
MAX_LR=0.0001
PATIENCE=10
EPOCHS=50
FOLD=1                     # Fixed fold 1
GPU="0"                  # GPU device numbers (multiple GPUs) - 修改为支持多GPU
MATCHED=true
USE_DEMOGRAPHICS=false
CROSS_EVAL=""              # Set to "matched_to_full" or "full_to_matched" if needed
PRETRAINED=true

# HealNet-specific parameters
DEPTH=1                    # Number of fusion layers
LATENT_CHANNELS=256        # Number of latent tokens (l_c)
LATENT_DIM=256             # Dimension of latent tokens (l_d)
CROSS_HEADS=4              # Number of cross-attention heads
LATENT_HEADS=4             # Number of self-attention heads
CROSS_DIM_HEAD=64          # Dimension of each cross-attention head
LATENT_DIM_HEAD=64         # Dimension of each self-attention head
SELF_PER_CROSS_ATTN=1      # Self-attention layers per cross-attention
WEIGHT_TIE_LAYERS=true     # Whether to share weights across layers
SNN=true                   # Whether to use self-normalizing networks
FOURIER_ENCODE_DATA=true   # Whether to use Fourier positional encoding
NUM_FREQ_BANDS=4           # Number of frequency bands
MAX_FREQ=5.0              # Maximum frequency for encoding
FINAL_CLASSIFIER_HEAD=true # Whether to add final classification head
ATTN_DROPOUT=0.2           # Dropout rate for attention layers
FF_DROPOUT=0.2             # Dropout rate for feed-forward layers

# Tasks configuration
TASK=("los")

# Set task-specific parameters
if [ "$TASK" = "phenotype" ]; then
    NUM_CLASSES=25
    INPUT_DIM=49          # Note: HealNet uses 49, not 498
elif [ "$TASK" = "mortality" ]; then
    NUM_CLASSES=1
    INPUT_DIM=49
else
    echo "Error: Unknown task $TASK"
    exit 1
fi

echo "Task: $TASK"
echo "Number of classes: $NUM_CLASSES"
echo "Input dimension: $INPUT_DIM"

# =====================================================================
# Experiment Configuration
# =====================================================================

# CXR dropout rates and random seeds
DROPOUT_RATES=(0.1 0.3 0.5 0.7 0.9)  # 20%, 40%, 60%, 80%
SEEDS=(42 123 456)

# Experiment directory - include task in directory name
DROPOUT_STR=$(printf "%s" "${DROPOUT_RATES[@]}" | sed 's/ /_/g')
BASE_EXP_DIR="../experiments_dropout/healnet_${TASK}_cxr_dropout_${DROPOUT_STR}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_EXP_DIR/logs"
mkdir -p "$BASE_EXP_DIR" "$LOG_DIR"

# Create main experiment log
MAIN_LOG="$LOG_DIR/experiment_main.log"

# Function to log with timestamp
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_msg "Starting HealNet $TASK CXR Dropout Experiment"
log_msg "Task: $TASK (classes: $NUM_CLASSES, input_dim: $INPUT_DIM)"
log_msg "CXR Dropout rates: ${DROPOUT_RATES[*]}"
log_msg "Random seeds: ${SEEDS[*]}"
log_msg "Results will be saved to: $BASE_EXP_DIR"

# =====================================================================
# Experiment Execution
# =====================================================================

# Calculate total number of experiments
total_experiments=$((${#DROPOUT_RATES[@]} * ${#SEEDS[@]}))
current_experiment=0

log_msg "Total number of experiments: $total_experiments"

# Start experiments
for DROPOUT_RATE in "${DROPOUT_RATES[@]}"; do
    if [ "$DROPOUT_RATE" = "0.2" ]; then
        DROPOUT_PERCENT=20
    elif [ "$DROPOUT_RATE" = "0.4" ]; then
        DROPOUT_PERCENT=40
    elif [ "$DROPOUT_RATE" = "0.6" ]; then
        DROPOUT_PERCENT=60
    elif [ "$DROPOUT_RATE" = "0.8" ]; then
        DROPOUT_PERCENT=80
    else
        DROPOUT_PERCENT=$(python3 -c "print(int(float('$DROPOUT_RATE') * 100))")
    fi
    
    DROPOUT_DIR="$BASE_EXP_DIR/dropout_${DROPOUT_PERCENT}percent"
    mkdir -p "$DROPOUT_DIR"
    
    log_msg "========================================================"
    log_msg "CXR Dropout Rate: ${DROPOUT_PERCENT}%"
    log_msg "Dropout directory: $DROPOUT_DIR"
    log_msg "========================================================"
    
    for SEED in "${SEEDS[@]}"; do
        ((current_experiment++))
        
        log_msg "Starting experiment $current_experiment/$total_experiments: Seed $SEED"
        
        # Create experiment directory - include task in experiment name
        EXP_NAME="healnet_${TASK}_dropout${DROPOUT_PERCENT}_seed${SEED}"
        EXP_DIR="$DROPOUT_DIR/$EXP_NAME"
        mkdir -p "$EXP_DIR"
        
        # Create experiment-specific log
        EXP_LOG="$LOG_DIR/${EXP_NAME}.log"
        
        log_msg "Experiment directory: $EXP_DIR"
        log_msg "Experiment log: $EXP_LOG"
        log_msg "Starting training..."
        
        # Build training command
        CMD="python ../main.py \
            --model healnet \
            --mode train \
            --task $TASK \
            --fold $FOLD \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --patience $PATIENCE \
            --epochs $EPOCHS \
            --seed $SEED \
            --input_dim $INPUT_DIM \
            --num_classes $NUM_CLASSES \
            --depth $DEPTH \
            --latent_channels $LATENT_CHANNELS \
            --latent_dim $LATENT_DIM \
            --cross_heads $CROSS_HEADS \
            --latent_heads $LATENT_HEADS \
            --cross_dim_head $CROSS_DIM_HEAD \
            --latent_dim_head $LATENT_DIM_HEAD \
            --self_per_cross_attn $SELF_PER_CROSS_ATTN \
            --attn_dropout $ATTN_DROPOUT \
            --ff_dropout $FF_DROPOUT \
            --num_freq_bands $NUM_FREQ_BANDS \
            --max_freq $MAX_FREQ \
            --cxr_dropout_rate $DROPOUT_RATE \
            --cxr_dropout_seed $SEED \
            --log_dir $EXP_DIR \
            --gpu $GPU"
        
        # Add conditional parameters
        if [ "$PRETRAINED" = "true" ]; then
            CMD="$CMD --pretrained"
        fi
        
        if [ "$WEIGHT_TIE_LAYERS" = "true" ]; then
            CMD="$CMD --weight_tie_layers"
        fi
        
        if [ "$SNN" = "true" ]; then
            CMD="$CMD --snn"
        fi
        
        if [ "$FOURIER_ENCODE_DATA" = "true" ]; then
            CMD="$CMD --fourier_encode_data"
        fi
        
        if [ "$FINAL_CLASSIFIER_HEAD" = "true" ]; then
            CMD="$CMD --final_classifier_head"
        fi
        
        if [ "$MATCHED" = "true" ]; then
            CMD="$CMD --matched"
        fi
        
        if [ "$USE_DEMOGRAPHICS" = "true" ]; then
            CMD="$CMD --use_demographics"
        fi
        
        # Log the command
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Command: $CMD" >> "$EXP_LOG"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting training for $EXP_NAME" >> "$EXP_LOG"
        
        # Execute training and redirect output to experiment log
        eval $CMD >> "$EXP_LOG" 2>&1
        
        # Check results
        if [ $? -eq 0 ]; then
            log_msg "Completed successfully: $EXP_NAME"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Training completed successfully" >> "$EXP_LOG"
        else
            log_msg "Failed: $EXP_NAME"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Training failed" >> "$EXP_LOG"
        fi
        
        log_msg "Progress: $current_experiment/$total_experiments"
        
        # Brief pause between experiments
        sleep 2
    done
done

# =====================================================================
# Final Summary
# =====================================================================

log_msg ""
log_msg "========================================================"
log_msg "All experiments completed!"
log_msg "========================================================"
log_msg "HealNet $TASK CXR Dropout Experiment Results:"
log_msg "Task: $TASK"
log_msg "Total experiments: $total_experiments"
log_msg "CXR Dropout rates: ${DROPOUT_RATES[*]}"
log_msg "Random seeds: ${SEEDS[*]}"
log_msg ""
log_msg "All results saved in: $BASE_EXP_DIR"
log_msg "Main log: $MAIN_LOG"
log_msg "Individual experiment logs: $LOG_DIR/"
log_msg ""
log_msg "To view results:"
log_msg "cd $BASE_EXP_DIR"
log_msg "ls -la"
log_msg "========================================================"

# Create a summary file
SUMMARY_FILE="$BASE_EXP_DIR/experiment_summary.txt"
cat > "$SUMMARY_FILE" << EOF
HealNet $TASK CXR Dropout Robustness Experiment Summary
=========================================================

Experiment Details:
- Model: HealNet
- Task: $TASK
- Number of classes: $NUM_CLASSES
- Input dimension: $INPUT_DIM
- Total experiments: $total_experiments
- CXR dropout rates: ${DROPOUT_RATES[*]}
- Random seeds: ${SEEDS[*]}
- Start time: $(date)

Directory Structure:
- Base directory: $BASE_EXP_DIR
- Logs directory: $LOG_DIR
- Main log: $MAIN_LOG
- Dropout-specific directories: dropout_*percent/
- Individual experiment directories: healnet_${TASK}_dropout*_seed*

HealNet Parameters Used:
- Batch size: $BATCH_SIZE
- Learning rate: $LR
- Epochs: $EPOCHS
- Patience: $PATIENCE
- Depth: $DEPTH
- Latent channels: $LATENT_CHANNELS
- Latent dimension: $LATENT_DIM
- Cross heads: $CROSS_HEADS
- Latent heads: $LATENT_HEADS

To analyze results:
1. Check individual experiment logs in $LOG_DIR/
2. Look for test_set_results.yaml in each experiment directory
3. Compare performance metrics across different dropout rates
4. Results are organized by dropout rate in separate subdirectories
EOF

log_msg "Experiment summary saved to: $SUMMARY_FILE"