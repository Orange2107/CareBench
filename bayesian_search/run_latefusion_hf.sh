#!/bin/bash

# ============================================================================
# LATEFUSION FIXED RUN (NO SEARCH)
# - Task: phenotype (9 labels via --use_phenotype9)
# - Data: full set (matched=false)
# - Seeds: 42, 123, 1234
# - CXR Encoder: hf_chexpert_vit (fixed)
# ============================================================================

set -u

# -------------------------
# User-configurable settings
# -------------------------

"""
python main.py \
  --model latefusion \
  --mode train \
  --task phenotype \
  --fold 1 \
  --gpu 0 \
  --lr 0.0001 \
  --batch_size 16 \
  --epochs 50 \
  --patience 10 \
  --num_classes 9 \
  --input_dim 49 \
  --ehr_encoder transformer \
  --hidden_size 256 \
  --ehr_dropout 0.2 \
  --ehr_n_layers 1 \
  --ehr_n_head 4 \
  --cxr_encoder hf_chexpert_vit \
  --hf_model_id codewithdark/vit-chest-xray \
  --partial_layers 0 \
  --pretrained \
  --use_phenotype9 \
  --freeze_vit true \
  --bias_tune false \
  --log_dir /hdd/bayesian_search_experiments/latefusion/phenotype
"""


MODEL="latefusion"
TASK="phenotype"
FOLD=1
GPU="0,1,2"                 # round-robin assignment across seeds
SEEDS=(42 123 1234)

# Fixed training setup
EPOCHS=50
PATIENCE=10
LR=0.0001
BATCH_SIZE=16
PRETRAINED=true
MATCHED=false               # full set
USE_DEMOGRAPHICS=false

# Phenotype-9 setup
NUM_CLASSES=9
USE_PHENOTYPE9=true

# LateFusion fixed architecture
EHR_ENCODER="transformer"
HIDDEN_SIZE=256
EHR_DROPOUT=0.2
EHR_N_LAYERS=1
EHR_N_HEAD=4
INPUT_DIM=49

# Fixed CXR encoder config (HF ViT)
CXR_ENCODER="hf_chexpert_vit"
HF_MODEL_ID="codewithdark/vit-chest-xray"
FREEZE_VIT=true
BIAS_TUNE=false
PARTIAL_LAYERS=0

# -------------------------
# Paths and logging
# -------------------------
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/hdd/bayesian_search_experiments/${MODEL}/${TASK}/lightning_logs/${MODEL}_${TASK}_phenotype9_full_fixed_run"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_DIR}/run_${RUN_TAG}.log"

mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_single_seed() {
    local seed="$1"
    local gpu_id="$2"
    local seed_dir="$3"
    local out_log="${seed_dir}/output.log"

    mkdir -p "$seed_dir"

    local cmd=(
        python main.py
        --model "$MODEL"
        --mode train
        --task "$TASK"
        --fold "$FOLD"
        --gpu "$gpu_id"
        --seed "$seed"
        --epochs "$EPOCHS"
        --patience "$PATIENCE"
        --lr "$LR"
        --batch_size "$BATCH_SIZE"
        --num_classes "$NUM_CLASSES"
        --input_dim "$INPUT_DIM"
        --ehr_encoder "$EHR_ENCODER"
        --hidden_size "$HIDDEN_SIZE"
        --ehr_dropout "$EHR_DROPOUT"
        --ehr_n_layers "$EHR_N_LAYERS"
        --ehr_n_head "$EHR_N_HEAD"
        --cxr_encoder "$CXR_ENCODER"
        --hf_model_id "$HF_MODEL_ID"
        --partial_layers "$PARTIAL_LAYERS"
        --log_dir "/hdd/bayesian_search_experiments/${MODEL}/${TASK}"
    )

    if [ "$PRETRAINED" = true ]; then
        cmd+=(--pretrained)
    fi
    if [ "$MATCHED" = true ]; then
        cmd+=(--matched)
    fi
    if [ "$USE_DEMOGRAPHICS" = true ]; then
        cmd+=(--use_demographics)
    fi
    if [ "$USE_PHENOTYPE9" = true ]; then
        cmd+=(--use_phenotype9)
    fi
    cmd+=(--freeze_vit "$FREEZE_VIT")
    cmd+=(--bias_tune "$BIAS_TUNE")

    log "Seed ${seed}: start on GPU ${gpu_id}"
    (
        cd "$BASE_DIR/.."
        "${cmd[@]}"
    ) >"$out_log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        log "Seed ${seed}: completed"
    else
        log "Seed ${seed}: failed (exit=${rc})"
    fi
    return $rc
}

main() {
    log "Starting fixed LateFusion run: phenotype9 + full set + 3 seeds"
    log "Results dir: $RESULTS_DIR"
    log "Run tag: $RUN_TAG"

    IFS=',' read -r -a GPU_LIST <<< "$GPU"
    local num_gpus="${#GPU_LIST[@]}"
    if [ "$num_gpus" -eq 0 ]; then
        log "No GPU configured. Set GPU=\"0\" or similar."
        exit 1
    fi

    # Launch all seeds in parallel (round-robin GPU assignment)
    local pids=()
    local seeds_order=()
    for i in "${!SEEDS[@]}"; do
        seed="${SEEDS[$i]}"
        gpu_id="${GPU_LIST[$((i % num_gpus))]}"
        seed_dir="${RESULTS_DIR}/seed${seed}_${RUN_TAG}"
        run_single_seed "$seed" "$gpu_id" "$seed_dir" &
        pids+=($!)
        seeds_order+=("$seed")
        log "Launched seed ${seed} (pid=${pids[-1]}) on GPU ${gpu_id}"
    done

    # Wait and collect status
    local fail_count=0
    for i in "${!pids[@]}"; do
        pid="${pids[$i]}"
        seed="${seeds_order[$i]}"
        if wait "$pid"; then
            log "Seed ${seed}: process finished successfully"
        else
            log "Seed ${seed}: process finished with failure"
            fail_count=$((fail_count + 1))
        fi
    done

    if [ "$fail_count" -eq 0 ]; then
        log "All seeds completed successfully."
        exit 0
    else
        log "${fail_count} seed run(s) failed. Check seed output logs in: $RESULTS_DIR"
        exit 1
    fi
}

trap 'log "Interrupted by user"; exit 1' SIGINT SIGTERM
main "$@"

