#!/bin/bash

# =====================================================================
# UMSE Phenotype CXR Dropout Robustness Experiment
# =====================================================================
# 
# Test UMSE model performance on phenotype task with different CXR dropout rates
# Uses the best parameters from Bayesian optimization (matched dataset)
# =====================================================================

echo "========================================================"
echo "UMSE Phenotype CXR Dropout Experiment"
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
GPU=3
MATCHED=true
USE_DEMOGRAPHICS=false
USE_TRIPLET=true
PRETRAINED=true

# UMSE specific parameters (from configs/train_configs/umse_phenotype.yaml - matched)
D_MODEL=256
VARIABLES_NUM=49
NUM_LAYERS=1
NUM_HEADS=4
N_MODALITY=2
BOTTLENECKS_N=1
MAX_EHR_LEN=500
DROPOUT=0.2

# Phenotype task parameters
TASK="phenotype"
NUM_CLASSES=25
INPUT_DIM=49

# =====================================================================
# Experiment Configuration
# =====================================================================

# CXR dropout rates and random seeds
DROPOUT_RATES=(0.2 0.4 0.6 0.8)  # 20%, 40%, 60%, 80%
SEEDS=(42 123 456)

# Create base experiment directory including all dropout rates
DROPOUT_STR=$(printf "%s" "${DROPOUT_RATES[@]}" | sed 's/ /_/g')
BASE_EXP_DIR="../experiments_robustness/umse_${TASK}_cxr_dropout_${DROPOUT_STR}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_EXP_DIR/logs"
mkdir -p "$BASE_EXP_DIR" "$LOG_DIR"

# Create main experiment log
MAIN_LOG="$LOG_DIR/experiment_main.log"

# Function to log with timestamp
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_msg "Starting UMSE Phenotype CXR Dropout Experiment"
log_msg "Task: $TASK"
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
    if [ "$DROPOUT_RATE" = "0.1" ]; then
        DROPOUT_PERCENT=10
    elif [ "$DROPOUT_RATE" = "0.2" ]; then
        DROPOUT_PERCENT=20
    elif [ "$DROPOUT_RATE" = "0.3" ]; then
        DROPOUT_PERCENT=30
    elif [ "$DROPOUT_RATE" = "0.4" ]; then
        DROPOUT_PERCENT=40
    elif [ "$DROPOUT_RATE" = "0.5" ]; then
        DROPOUT_PERCENT=50
    elif [ "$DROPOUT_RATE" = "0.6" ]; then
        DROPOUT_PERCENT=60
    elif [ "$DROPOUT_RATE" = "0.7" ]; then
        DROPOUT_PERCENT=70
    elif [ "$DROPOUT_RATE" = "0.8" ]; then
        DROPOUT_PERCENT=80
    elif [ "$DROPOUT_RATE" = "0.9" ]; then
        DROPOUT_PERCENT=90
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
        EXP_NAME="umse_phenotype_dropout${DROPOUT_PERCENT}_seed${SEED}"
        EXP_DIR="$DROPOUT_DIR/$EXP_NAME"
        mkdir -p "$EXP_DIR"
        
        # Create experiment-specific log
        EXP_LOG="$LOG_DIR/${EXP_NAME}.log"
        
        log_msg "Experiment directory: $EXP_DIR"
        log_msg "Experiment log: $EXP_LOG"
        log_msg "Starting training..."
        
        # Build training command
        CMD="python ../main.py \
            --model umse \
            --mode train \
            --task $TASK \
            --fold $FOLD \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --patience $PATIENCE \
            --epochs $EPOCHS \
            --seed $SEED \
            --d_model $D_MODEL \
            --variables_num $VARIABLES_NUM \
            --num_layers $NUM_LAYERS \
            --num_heads $NUM_HEADS \
            --n_modality $N_MODALITY \
            --bottlenecks_n $BOTTLENECKS_N \
            --max_ehr_len $MAX_EHR_LEN \
            --dropout $DROPOUT \
            --num_classes $NUM_CLASSES \
            --cxr_dropout_rate $DROPOUT_RATE \
            --cxr_dropout_seed $SEED \
            --log_dir $EXP_DIR \
            --use_label_weights \
            --compute_fairness \
            --save_predictions \
            --gpu $GPU"
        
        # Add conditional parameters
        if [ "$MATCHED" = "true" ]; then
            CMD="$CMD --matched"
        fi
        
        if [ "$USE_DEMOGRAPHICS" = "true" ]; then
            CMD="$CMD --use_demographics"
        fi
        
        if [ "$USE_TRIPLET" = "true" ]; then
            CMD="$CMD --use_triplet"
        fi
        
        if [ "$PRETRAINED" = "true" ]; then
            CMD="$CMD --pretrained"
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
log_msg "UMSE Phenotype CXR Dropout Experiment Results:"
log_msg "Task: $TASK"
log_msg "Total experiments: $total_experiments"
log_msg "CXR Dropout rates: ${DROPOUT_RATES[*]}"
log_msg "Random seeds: ${SEEDS[*]}"
log_msg ""
log_msg "All results saved in: $BASE_EXP_DIR"
log_msg "Main log: $MAIN_LOG"
log_msg "Individual experiment logs: $LOG_DIR/"
log_msg ""
log_msg "Directory structure:"
log_msg "- Base: $BASE_EXP_DIR"
for DROPOUT_RATE in "${DROPOUT_RATES[@]}"; do
    # Convert dropout rate to percentage using the same logic
    if [ "$DROPOUT_RATE" = "0.1" ]; then
        DROPOUT_PERCENT=10
    elif [ "$DROPOUT_RATE" = "0.2" ]; then
        DROPOUT_PERCENT=20
    elif [ "$DROPOUT_RATE" = "0.3" ]; then
        DROPOUT_PERCENT=30
    elif [ "$DROPOUT_RATE" = "0.4" ]; then
        DROPOUT_PERCENT=40
    elif [ "$DROPOUT_RATE" = "0.5" ]; then
        DROPOUT_PERCENT=50
    elif [ "$DROPOUT_RATE" = "0.6" ]; then
        DROPOUT_PERCENT=60
    elif [ "$DROPOUT_RATE" = "0.7" ]; then
        DROPOUT_PERCENT=70
    elif [ "$DROPOUT_RATE" = "0.8" ]; then
        DROPOUT_PERCENT=80
    elif [ "$DROPOUT_RATE" = "0.9" ]; then
        DROPOUT_PERCENT=90
    else
        DROPOUT_PERCENT=$(python3 -c "print(int(float('$DROPOUT_RATE') * 100))")
    fi
    log_msg "- Dropout ${DROPOUT_PERCENT}%: $BASE_EXP_DIR/dropout_${DROPOUT_PERCENT}percent/"
done
log_msg "========================================================"

# Create a summary file
SUMMARY_FILE="$BASE_EXP_DIR/experiment_summary.txt"
cat > "$SUMMARY_FILE" << EOF
UMSE Phenotype CXR Dropout Robustness Experiment Summary
=========================================================

Experiment Details:
- Model: UMSE
- Task: Phenotype classification
- Total experiments: $total_experiments
- CXR dropout rates: ${DROPOUT_RATES[*]}
- Random seeds: ${SEEDS[*]}
- Start time: $(date)

Directory Structure:
- Base directory: $BASE_EXP_DIR
- Logs directory: $LOG_DIR
- Main log: $MAIN_LOG
- Dropout-specific directories: dropout_*percent/
- Individual experiment directories: umse_phenotype_dropout*_seed*

Parameters Used:
- Batch size: $BATCH_SIZE
- Learning rate: $LR
- Epochs: $EPOCHS
- Patience: $PATIENCE
- D model: $D_MODEL
- Variables num: $VARIABLES_NUM
- Num layers: $NUM_LAYERS
- Num heads: $NUM_HEADS
- N modality: $N_MODALITY
- Bottlenecks n: $BOTTLENECKS_N
- Max EHR len: $MAX_EHR_LEN
- Dropout: $DROPOUT
- Use triplet: $USE_TRIPLET

To analyze results:
1. Check individual experiment logs in $LOG_DIR/
2. Look for test_set_results.yaml in each experiment directory
3. Compare performance metrics across different dropout rates
4. Results are organized by dropout rate in separate subdirectories
EOF

log_msg "Experiment summary saved to: $SUMMARY_FILE"
