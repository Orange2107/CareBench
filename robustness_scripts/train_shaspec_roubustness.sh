#!/bin/bash

# =====================================================================
# ShaSpec CXR Dropout Robustness Experiment
# =====================================================================
# 
# Test ShaSpec model performance with different CXR dropout rates
# Uses the same parameters as the original ShaSpec training script
# =====================================================================

echo "========================================================"
echo "ShaSpec CXR Dropout Experiment"
echo "========================================================"

# =====================================================================
# Basic Configuration
# =====================================================================

# Basic training parameters
BATCH_SIZE=16
LR=0.0001
PATIENCE=10
EPOCHS=50
FOLD=1
GPU=2
MATCHED=true
USE_DEMOGRAPHICS=false

# ShaSpec specific parameters (based on best parameters from Bayesian search)
EHR_ENCODER=transformer    # Options: 'lstm', 'transformer'
CXR_ENCODER=resnet50       # Options: 'resnet50'
DIM=256                    # Hidden feature dimension
DROPOUT=0.2              # Dropout rate
ALPHA=0.08168886881742098                # Shared feature consistency loss weight
BETA=0.040769784023901946                 # Domain classification loss weight
NHEAD=4                    # Number of attention heads for shared transformer
NUM_LAYERS=1               # Number of layers for shared transformer
MAX_SEQ_LEN=500            # Maximum sequence length

# EHR Encoder specific parameters
EHR_N_HEAD=4               # Number of attention heads for EHR transformer
EHR_N_LAYERS=1             # Number of layers for EHR transformer

# ShaSpec specific flags
WEIGHT_STD=true            # Use weight standardization
PRETRAINED=true            # Use pretrained CXR encoder

# Tasks configuration
TASK="los"

# Set task-specific parameters
if [ "$TASK" = "phenotype" ]; then
    NUM_CLASSES=25
    INPUT_DIM=49
elif [ "$TASK" = "mortality" ]; then
    NUM_CLASSES=1
    INPUT_DIM=49
elif [ "$TASK" = "los" ]; then
    NUM_CLASSES=7
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
DROPOUT_RATES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)  # 20%, 40%, 60%, 80%
SEEDS=(42 123 456)

# Create base experiment directory including all dropout rates
DROPOUT_STR=$(printf "%s" "${DROPOUT_RATES[@]}" | sed 's/ /_/g')
BASE_EXP_DIR="../experiments_dropout/shaspec_${TASK}_cxr_dropout_${DROPOUT_STR}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_EXP_DIR/logs"
mkdir -p "$BASE_EXP_DIR" "$LOG_DIR"

# Create main experiment log
MAIN_LOG="$LOG_DIR/experiment_main.log"

# Function to log with timestamp
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_msg "Starting ShaSpec $TASK CXR Dropout Experiment"
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
    # Convert dropout rate to percentage using pure bash arithmetic
    if [ "$DROPOUT_RATE" = "0.2" ]; then
        DROPOUT_PERCENT=20
    elif [ "$DROPOUT_RATE" = "0.4" ]; then
        DROPOUT_PERCENT=40
    elif [ "$DROPOUT_RATE" = "0.6" ]; then
        DROPOUT_PERCENT=60
    elif [ "$DROPOUT_RATE" = "0.8" ]; then
        DROPOUT_PERCENT=80
    else
        # Fallback: use python for calculation
        DROPOUT_PERCENT=$(python3 -c "print(int(float('$DROPOUT_RATE') * 100))")
    fi
    
    # Create independent subdirectory for each dropout rate
    DROPOUT_DIR="$BASE_EXP_DIR/dropout_${DROPOUT_PERCENT}percent"
    mkdir -p "$DROPOUT_DIR"
    
    log_msg "========================================================"
    log_msg "CXR Dropout Rate: ${DROPOUT_PERCENT}%"
    log_msg "Dropout directory: $DROPOUT_DIR"
    log_msg "========================================================"
    
    for SEED in "${SEEDS[@]}"; do
        ((current_experiment++))
        
        log_msg "Starting experiment $current_experiment/$total_experiments: Seed $SEED"
        
        # Create experiment directory within dropout-specific directory
        EXP_NAME="shaspec_${TASK}_dropout${DROPOUT_PERCENT}_seed${SEED}"
        EXP_DIR="$DROPOUT_DIR/$EXP_NAME"
        mkdir -p "$EXP_DIR"
        
        # Create experiment-specific log
        EXP_LOG="$LOG_DIR/${EXP_NAME}.log"
        
        log_msg "Experiment directory: $EXP_DIR"
        log_msg "Experiment log: $EXP_LOG"
        log_msg "Starting training..."
        
        # Build training command
        CMD="python ../main.py \
            --model shaspec \
            --mode train \
            --task $TASK \
            --fold $FOLD \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --patience $PATIENCE \
            --epochs $EPOCHS \
            --seed $SEED \
            --ehr_encoder $EHR_ENCODER \
            --cxr_encoder $CXR_ENCODER \
            --dim $DIM \
            --dropout $DROPOUT \
            --alpha $ALPHA \
            --beta $BETA \
            --nhead $NHEAD \
            --num_layers $NUM_LAYERS \
            --max_seq_len $MAX_SEQ_LEN \
            --ehr_n_head $EHR_N_HEAD \
            --ehr_n_layers $EHR_N_LAYERS \
            --input_dim $INPUT_DIM \
            --num_classes $NUM_CLASSES \
            --use_label_weights \
            --cxr_dropout_rate $DROPOUT_RATE \
            --cxr_dropout_seed $SEED \
            --log_dir $EXP_DIR \
            --gpu $GPU"
        
        # Add conditional parameters
        if [ "$PRETRAINED" = "true" ]; then
            CMD="$CMD --pretrained"
        fi
        
        if [ "$WEIGHT_STD" = "true" ]; then
            CMD="$CMD --weight_std"
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
log_msg "ShaSpec $TASK CXR Dropout Experiment Results:"
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
ShaSpec $TASK CXR Dropout Robustness Experiment Summary
=========================================================

Experiment Details:
- Model: ShaSpec
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
- Dropout subdirectories: dropout_20percent/, dropout_40percent/, etc.
- Individual experiment directories: shaspec_${TASK}_dropout*_seed*

ShaSpec Parameters Used:
- Batch size: $BATCH_SIZE
- Learning rate: $LR
- Epochs: $EPOCHS
- Patience: $PATIENCE
- EHR encoder: $EHR_ENCODER
- CXR encoder: $CXR_ENCODER
- Dimension: $DIM
- Alpha (consistency loss): $ALPHA
- Beta (domain loss): $BETA
- Number of heads: $NHEAD
- Number of layers: $NUM_LAYERS

To analyze results:
1. Check individual experiment logs in $LOG_DIR/
2. Look for test_set_results.yaml in each dropout subdirectory
3. Compare performance metrics across different dropout rates
EOF

log_msg "Experiment summary saved to: $SUMMARY_FILE"