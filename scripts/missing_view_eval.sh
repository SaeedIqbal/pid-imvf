#!/bin/bash

# =====================================================================
# PID-IMVF: missing_view_eval.sh
# Object-Oriented Bash Script for Missing-View Robustness Evaluation.
#
# Mathematical Background (Theorem 3):
# Under missing-view conditions, the classifier is strictly restricted 
# to the partial consensus shared representation:
#   \hat{S}^{(\text{obs})} = \frac{1}{|\mathcal{V}_{\text{obs}}|} \sum_{v \in \mathcal{V}_{\text{obs}}} \hat{S}_v
#
# The view-specific factors \hat{U}_v are explicitly excluded. 
# The degradation in Bayes error is bounded by:
#   \Delta P_e \leq \frac{1}{2} (U_k + S_k)
#
# To strictly avoid random synthetic evaluation, this script employs 
# deterministic dropping strategies to simulate exactly 50% (or the 
# closest integer approximation) view dropout for each dataset.
# =====================================================================

set -e

# =====================================================================
# Class: MissingViewEvaluator
# Encapsulates the state and methods for evaluating missing-view robustness.
# =====================================================================

# Static Variables (Class Attributes)
MissingViewEvaluator::DATASET_ROOT="/home/phd/datasets"
MissingViewEvaluator::LOG_ROOT="/home/phd/logs/missing_view"
MissingViewEvaluator::CHECKPOINT_ROOT="/home/phd/checkpoints"
MissingViewEvaluator::CONFIG_DIR="configs/datasets"

# Array of target datasets and their corresponding configs
MissingViewEvaluator::DATASETS=("100leaves" "animals" "hw" "mnist" "noisy_mnist" "uci")

# =====================================================================
# Method: init
# Initializes the environment and creates necessary directories.
# =====================================================================
MissingViewEvaluator::init() {
    echo "========================================================"
    echo "[MissingViewEvaluator] Initializing evaluation environment..."
    echo "Dataset Root: ${MissingViewEvaluator::DATASET_ROOT}"
    echo "Log Root: ${MissingViewEvaluator::LOG_ROOT}"
    echo "========================================================"
    
    mkdir -p "${MissingViewEvaluator::LOG_ROOT}"
}

# =====================================================================
# Method: check_full_view_checkpoint
# Verifies that the full-view model (trained via train_all.sh) exists.
# =====================================================================
MissingViewEvaluator::check_full_view_checkpoint() {
    local dataset=$1
    local checkpoint_path="${MissingViewEvaluator::CHECKPOINT_ROOT}/pid_imvf_${dataset}/epoch=49-step=*.ckpt"
    
    # Find the actual checkpoint file (handling PyTorch Lightning step naming)
    local actual_ckpt=$(ls ${checkpoint_path} 2>/dev/null | head -n 1)
    
    if [ -z "$actual_ckpt" ]; then
        echo "ERROR: Full-view checkpoint not found for ${dataset}."
        echo "Please run scripts/train_all.sh first."
        return 1
    fi
    
    echo "$actual_ckpt"
    return 0
}

# =====================================================================
# Method: get_deterministic_kept_views
# Returns a comma-separated list of view indices to KEEP to achieve 
# exactly 50% (or closest integer) deterministic dropout.
# NO RANDOM VALUES ARE USED.
#
# Mathematical mapping:
#   V=2 (Animals, NoisyMNIST) -> Keep 1 view (50% drop) -> "0"
#   V=3 (100leaves, MNIST, UCI) -> Keep 2 views (~33% drop) -> "0,1"
#   V=6 (HW) -> Keep 3 views (50% drop) -> "0,1,2"
# =====================================================================
MissingViewEvaluator::get_deterministic_kept_views() {
    local dataset=$1
    
    case "$dataset" in
        "animals"|"noisy_mnist")
            echo "0"
            ;;
        "100leaves"|"mnist"|"uci")
            echo "0,1"
            ;;
        "hw")
            echo "0,1,2"
            ;;
        *)
            echo "ERROR: Unknown dataset ${dataset} for deterministic masking." >&2
            return 1
            ;;
    esac
}

# =====================================================================
# Method: run_evaluation
# Executes the Python evaluation script for a specific dataset and 
# deterministic view mask.
# =====================================================================
MissingViewEvaluator::run_evaluation() {
    local dataset=$1
    local kept_views=$2
    local checkpoint=$3
    
    local config_file="${MissingViewEvaluator::CONFIG_DIR}/${dataset}.yaml"
    local log_file="${MissingViewEvaluator::LOG_ROOT}/eval_missing_view_${dataset}.log"
    
    echo "--------------------------------------------------------"
    echo "[MissingViewEvaluator] Evaluating: ${dataset}"
    echo "Checkpoint: ${checkpoint}"
    echo "Deterministic Kept Views: [${kept_views}]"
    echo "Log File: ${log_file}"
    echo "--------------------------------------------------------"
    
    # Execute the Python evaluation script
    # The evaluate.py script will internally call MissingViewSimulator 
    # to compute \hat{S}^{(\text{obs})} and evaluate ACC, Macro-F1, BACC.
    python evaluate.py \
        --dataset_config "$config_file" \
        --checkpoint "$checkpoint" \
        --dataset_root "${MissingViewEvaluator::DATASET_ROOT}" \
        --kept_views "$kept_views" \
        --log_dir "${MissingViewEvaluator::LOG_ROOT}" \
        --experiment_name "missing_view_${dataset}" \
        > "$log_file" 2>&1
        
    return $?
}

# =====================================================================
# Main Execution Block
# =====================================================================

# 1. Initialize Environment
MissingViewEvaluator::init

# 2. Iterate over all datasets
for dataset in "${MissingViewEvaluator::DATASETS[@]}"; do
    
    # 2.1 Check if the full-view model exists
    set +e
    checkpoint=$(MissingViewEvaluator::check_full_view_checkpoint "$dataset")
    ckpt_status=$?
    set -e
    
    if [ $ckpt_status -ne 0 ]; then
        echo "WARNING: Skipping ${dataset} due to missing full-view checkpoint."
        continue
    fi
    
    # 2.2 Get the deterministic 50% dropout mask
    kept_views=$(MissingViewEvaluator::get_deterministic_kept_views "$dataset")
    
    # 2.3 Run the evaluation
    set +e
    MissingViewEvaluator::run_evaluation "$dataset" "$kept_views" "$checkpoint"
    eval_status=$?
    set -e
    
    if [ $eval_status -eq 0 ]; then
        echo "SUCCESS: Completed missing-view evaluation for ${dataset}."
    else
        echo "WARNING: Evaluation failed for ${dataset}. Check ${log_file}."
    fi
    echo ""
done

echo "========================================================"
echo "[MissingViewEvaluator] All missing-view evaluations completed."
echo "Results and PID atom logs are available in: ${MissingViewEvaluator::LOG_ROOT}"
echo "========================================================"