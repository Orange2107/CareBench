#!/bin/bash

# =====================================================================
# AUG Phenotype CXR Dropout Robustness Experiment
# =====================================================================
# 
# Test AUG model performance on phenotype task with different CXR dropout rates
# Uses the best parameters from Bayesian optimization
# =====================================================================

echo "========================================================"
echo "AUG Phenotype CXR Dropout Experiment"
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
GPU=1
MATCHED=true
USE_DEMOGRAPHICS=false

# AUG specific parameters (from best_params.txt)
EHR_ENCODER=transformer
CXR_ENCODER=resnet50
HIDDEN_SIZE=256
EHR_DROPOUT=0.2
EHR_N_HEAD=4
EHR_N_LAYERS=1
EHR_NUM_LAYERS=1
EHR_BIDIRECTIONAL=true
PRETRAINED=true

# AUG-specific fusion parameters (from best_params.txt)
AUG_MERGE_ALPHA=0.6483920660824287
AUG_LAMBDA=1.015876852955632
AUG_MARGIN=0.031403802454873175
AUG_LAYER_CHECK_INTERVAL=15
AUG_CLASSIFIER_HIDDEN_DIM=256

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
BASE_EXP_DIR="../experiments_robustness/aug_${TASK}_cxr_dropout_${DROPOUT_STR}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_EXP_DIR/logs"
mkdir -p "$BASE_EXP_DIR" "$LOG_DIR"

# Create main experiment log
MAIN_LOG="$LOG_DIR/experiment_main.log"

# Function to log with timestamp
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_msg "Starting AUG Phenotype CXR Dropout Experiment"
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
        EXP_NAME="aug_phenotype_dropout${DROPOUT_PERCENT}_seed${SEED}"
        EXP_DIR="$DROPOUT_DIR/$EXP_NAME"
        mkdir -p "$EXP_DIR"
        
        # Create experiment-specific log
        EXP_LOG="$LOG_DIR/${EXP_NAME}.log"
        
        log_msg "Experiment directory: $EXP_DIR"
        log_msg "Experiment log: $EXP_LOG"
        log_msg "Starting training..."
        
        # Build training command
        CMD="python ../main.py \
            --model aug \
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
            --hidden_size $HIDDEN_SIZE \
            --ehr_dropout $EHR_DROPOUT \
            --ehr_n_head $EHR_N_HEAD \
            --ehr_n_layers $EHR_N_LAYERS \
            --ehr_num_layers $EHR_NUM_LAYERS \
            --input_dim $INPUT_DIM \
            --num_classes $NUM_CLASSES \
            --aug_merge_alpha $AUG_MERGE_ALPHA \
            --aug_lambda $AUG_LAMBDA \
            --aug_margin $AUG_MARGIN \
            --aug_layer_check_interval $AUG_LAYER_CHECK_INTERVAL \
            --aug_classifier_hidden_dim $AUG_CLASSIFIER_HIDDEN_DIM \
            --cxr_dropout_rate $DROPOUT_RATE \
            --cxr_dropout_seed $SEED \
            --log_dir $EXP_DIR \
            --use_label_weights \
            --compute_fairness \
            --save_predictions \
            --gpu $GPU"
        
        # Add conditional parameters
        if [ "$PRETRAINED" = "true" ]; then
            CMD="$CMD --pretrained"
        fi
        
        if [ "$EHR_BIDIRECTIONAL" = "true" ]; then
            CMD="$CMD --ehr_bidirectional"
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
log_msg "AUG Phenotype CXR Dropout Experiment Results:"
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
AUG Phenotype CXR Dropout Robustness Experiment Summary
=========================================================

Experiment Details:
- Model: AUG
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
- Individual experiment directories: aug_phenotype_dropout*_seed*

Parameters Used:
- Batch size: $BATCH_SIZE
- Learning rate: $LR
- Epochs: $EPOCHS
- Patience: $PATIENCE
- EHR encoder: $EHR_ENCODER
- CXR encoder: $CXR_ENCODER
- Hidden size: $HIDDEN_SIZE
- AUG merge alpha: $AUG_MERGE_ALPHA
- AUG lambda: $AUG_LAMBDA
- AUG margin: $AUG_MARGIN
- AUG layer check interval: $AUG_LAYER_CHECK_INTERVAL
- AUG classifier hidden dim: $AUG_CLASSIFIER_HIDDEN_DIM

To analyze results:
1. Check individual experiment logs in $LOG_DIR/
2. Look for test_set_results.yaml in each experiment directory
3. Compare performance metrics across different dropout rates
4. Results are organized by dropout rate in separate subdirectories
EOF

log_msg "Experiment summary saved to: $SUMMARY_FILE"
