#!/bin/bash

# =====================================================================
# SMIL CXR Dropout Robustness Experiment
# =====================================================================
# 
# Test SMIL model performance with different CXR dropout rates
# Uses the same parameters as the original SMIL training script
# =====================================================================

echo "========================================================"
echo "SMIL CXR Dropout Experiment"
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
GPU=4
MATCHED=true
USE_DEMOGRAPHICS=false
CROSS_EVAL=""
PRETRAINED=true

# SMIL-specific parameters
EHR_ENCODER="transformer"         # Options: "lstm", "transformer" 
CXR_ENCODER="resnet50"     # Options: "resnet50"
HIDDEN_DIM=256             # Hidden dimension for SMIL
INNER_LOOP=3               # Number of inner loops for meta-learning
LR_INNER=0.00048099463193341493     # Inner learning rate
MC_SIZE=10                 # Monte Carlo size
ALPHA=0.10998628646125844                 # Feature distillation weight
BETA=0.17555899317323298   # EHR mean distillation weight
TEMPERATURE=1.0842555683358894            # Knowledge distillation temperature
N_CLUSTERS=10              # Number of clusters for CXR k-means

# EHR LSTM parameters (used when EHR_ENCODER="lstm")
EHR_NUM_LAYERS=1
EHR_BIDIRECTIONAL=true

# EHR Transformer parameters (used when EHR_ENCODER="transformer")
EHR_N_HEAD=4
EHR_N_LAYERS=1
MAX_LEN=500

# Tasks configuration
TASK="los"

# Set task-specific parameters
if [ "$TASK" = "phenotype" ]; then
    NUM_CLASSES=25
    INPUT_DIM=49          # Note: SMIL uses 49 for this config
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
# Experiment Configuration - Same as DrFuse
# =====================================================================

# CXR dropout rates and random seeds
DROPOUT_RATES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)  
SEEDS=(42 123 456)

# Create base experiment directory including all dropout rates (same as DrFuse)
DROPOUT_STR=$(printf "%s" "${DROPOUT_RATES[@]}" | sed 's/ /_/g')
BASE_EXP_DIR="../experiments_dropout/smil_${TASK}_cxr_dropout_${DROPOUT_STR}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_EXP_DIR/logs"
mkdir -p "$BASE_EXP_DIR" "$LOG_DIR"

# Create main experiment log
MAIN_LOG="$LOG_DIR/experiment_main.log"

# Function to log with timestamp
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_msg "Starting SMIL $TASK CXR Dropout Experiment"
log_msg "Task: $TASK"
log_msg "CXR Dropout rates: ${DROPOUT_RATES[*]}"
log_msg "Random seeds: ${SEEDS[*]}"
log_msg "Results will be saved to: $BASE_EXP_DIR"

# Check if CXR k-means centers are available
DATA_TYPE="matched"
if [ "$MATCHED" = "false" ]; then
    DATA_TYPE="full"
fi

KMEANS_CENTERS_FILE="../data/cxr_kmeans_centers_${DATA_TYPE}_${TASK}_${N_CLUSTERS}.pkl"
if [ ! -f "$KMEANS_CENTERS_FILE" ]; then
    log_msg "Warning: CXR k-means centers file not found: $KMEANS_CENTERS_FILE"
    log_msg "SMIL model may not work properly without pre-computed k-means centers"
fi

# =====================================================================
# Experiment Execution
# =====================================================================

# Calculate total number of experiments
total_experiments=$((${#DROPOUT_RATES[@]} * ${#SEEDS[@]}))
current_experiment=0

log_msg "Total number of experiments: $total_experiments"

# Start experiments
for DROPOUT_RATE in "${DROPOUT_RATES[@]}"; do
    # Convert dropout rate to percentage using pure bash arithmetic (same as DrFuse)
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
    
    # Create independent subdirectory for each dropout rate (same as DrFuse)
    DROPOUT_DIR="$BASE_EXP_DIR/dropout_${DROPOUT_PERCENT}percent"
    mkdir -p "$DROPOUT_DIR"
    
    log_msg "========================================================"
    log_msg "CXR Dropout Rate: ${DROPOUT_PERCENT}%"
    log_msg "Dropout directory: $DROPOUT_DIR"
    log_msg "========================================================"
    
    for SEED in "${SEEDS[@]}"; do
        ((current_experiment++))
        
        log_msg "Starting experiment $current_experiment/$total_experiments: Seed $SEED"
        
        # Create experiment directory within dropout-specific directory (same as DrFuse)
        EXP_NAME="smil_${TASK}_dropout${DROPOUT_PERCENT}_seed${SEED}"
        EXP_DIR="$DROPOUT_DIR/$EXP_NAME"
        mkdir -p "$EXP_DIR"
        
        # Create experiment-specific log
        EXP_LOG="$LOG_DIR/${EXP_NAME}.log"
        
        log_msg "Experiment directory: $EXP_DIR"
        log_msg "Experiment log: $EXP_LOG"
        log_msg "Starting training..."
        
        # Build base command
        CMD="python ../main.py \
            --model smil \
            --fusion_type smil \
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
            --hidden_dim $HIDDEN_DIM \
            --inner_loop $INNER_LOOP \
            --lr_inner $LR_INNER \
            --mc_size $MC_SIZE \
            --alpha $ALPHA \
            --beta $BETA \
            --temperature $TEMPERATURE \
            --n_clusters $N_CLUSTERS \
            --input_dim $INPUT_DIM \
            --num_classes $NUM_CLASSES \
            --cxr_dropout_rate $DROPOUT_RATE \
            --cxr_dropout_seed $SEED \
            --log_dir $EXP_DIR \
            --gpu $GPU"
        
        # Add EHR encoder specific parameters
        if [ "$EHR_ENCODER" = "lstm" ]; then
            CMD="$CMD --ehr_num_layers $EHR_NUM_LAYERS"
            if [ "$EHR_BIDIRECTIONAL" = "true" ]; then
                CMD="$CMD --ehr_bidirectional"
            fi
        elif [ "$EHR_ENCODER" = "transformer" ]; then
            CMD="$CMD --ehr_n_head $EHR_N_HEAD --ehr_n_layers $EHR_N_LAYERS --max_len $MAX_LEN"
        fi
        
        # Add conditional parameters
        if [ "$PRETRAINED" = "true" ]; then
            CMD="$CMD --pretrained"
        fi
        
        if [ "$MATCHED" = "true" ]; then
            CMD="$CMD --matched"
        fi
        
        if [ "$USE_DEMOGRAPHICS" = "true" ]; then
            CMD="$CMD --use_demographics"
        fi
        
        if [ -n "$CROSS_EVAL" ]; then
            CMD="$CMD --cross_eval $CROSS_EVAL"
        fi
        
        # Log the command
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Command: $CMD" >> "$EXP_LOG"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting training for $EXP_NAME" >> "$EXP_LOG"
        
        log_msg "Running command: $CMD"
        log_msg "Logging to: $EXP_LOG"
        
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
    
    log_msg "All seeds completed for CXR dropout rate $DROPOUT_RATE"
done

log_msg "All CXR dropout rates completed for task $TASK"

# =====================================================================
# Final Summary
# =====================================================================

log_msg ""
log_msg "========================================================"
log_msg "All experiments completed!"
log_msg "========================================================"
log_msg "SMIL $TASK CXR Dropout Experiment Results:"
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
SMIL $TASK CXR Dropout Robustness Experiment Summary
=========================================================

Experiment Details:
- Model: SMIL
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
- Individual experiment directories: smil_${TASK}_dropout*_seed*

SMIL Parameters Used:
- Batch size: $BATCH_SIZE
- Learning rate: $LR
- Epochs: $EPOCHS
- Patience: $PATIENCE
- EHR encoder: $EHR_ENCODER
- CXR encoder: $CXR_ENCODER
- Hidden dimension: $HIDDEN_DIM
- Inner loop: $INNER_LOOP
- LR inner: $LR_INNER
- MC size: $MC_SIZE
- Alpha: $ALPHA
- Beta: $BETA
- Temperature: $TEMPERATURE
- N clusters: $N_CLUSTERS

To analyze results:
1. Check individual experiment logs in $LOG_DIR/
2. Look for test_set_results.yaml in each dropout_*percent/smil_*/ directory
3. Compare performance metrics across different dropout rates
EOF

log_msg "Experiment summary saved to: $SUMMARY_FILE"