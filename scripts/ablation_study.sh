#!/bin/bash

# =====================================================================
# PID-IMVF: ablation_study.sh
# Executes the 5 controlled ablation experiments to isolate the 
# contribution of each component in the proposed methodology.
# 
# Ablations:
# 1. w/o Redundancy (w_o_red)       -> Removes the Redundancy PID atom
# 2. w/o Uniqueness (w_o_unq)       -> Removes the Uniqueness PID atoms
# 3. w/o Synergy (w_o_syn)          -> Removes the Synergy PID atom
# 4. MLP Encoder (mlp_encoder)      -> Replaces Factorized INN with standard MLP
# 5. Concat Classifier (concat)     -> Classifier uses S_hat \oplus U_v instead of S_hat
# =====================================================================

# Exit immediately if a command exits with a non-zero status (unless handled)
set -e

# Define root directories for data, logs, and checkpoints
DATASET_ROOT="/home/phd/datasets"
LOG_ROOT="/home/phd/logs/ablations"
CHECKPOINT_ROOT="/home/phd/checkpoints/ablations"
BASE_DATASET_CONFIG_DIR="configs/datasets"
ABLATION_CONFIG_DIR="configs/ablations"

# Create necessary directories if they don't exist
mkdir -p "$LOG_ROOT"
mkdir -p "$CHECKPOINT_ROOT"

# Define the 5 ablation experiments
ABLATIONS=(
    "w_o_red"
    "w_o_unq"
    "w_o_syn"
    "mlp_encoder"
    "concat_classifier"
)

# Define target datasets for the ablation study 
# (HW to highlight synergy isolation, Animals to highlight class-imbalance resolution)
TARGET_DATASETS=(
    "hw"
    "animals"
)

# =====================================================================
# Modular Functions (Bash OOP equivalent)
# =====================================================================

# Function to check if required configuration files exist
check_dependencies() {
    local dataset=$1
    local ablation=$2
    local dataset_config="${BASE_DATASET_CONFIG_DIR}/${dataset}.yaml"
    local ablation_config="${ABLATION_CONFIG_DIR}/${ablation}.yaml"
    
    if [ ! -f "$dataset_config" ]; then
        echo "ERROR: Dataset config not found: $dataset_config"
        return 1
    fi
    
    if [ ! -f "$ablation_config" ]; then
        echo "ERROR: Ablation config not found: $ablation_config"
        return 1
    fi
    return 0
}

# Function to execute a single ablation run
run_ablation() {
    local dataset=$1
    local ablation=$2
    
    local dataset_config="${BASE_DATASET_CONFIG_DIR}/${dataset}.yaml"
    local ablation_config="${ABLATION_CONFIG_DIR}/${ablation}.yaml"
    local log_file="${LOG_ROOT}/${ablation}_${dataset}.log"
    local exp_name="ablation_${ablation}_${dataset}"
    
    echo "--------------------------------------------------------"
    echo "Executing Ablation: $ablation on Dataset: $dataset"
    echo "Dataset Config: $dataset_config"
    echo "Ablation Config: $ablation_config"
    echo "Log File: $log_file"
    echo "--------------------------------------------------------"
    
    # Run the main training script
    # The train.py script is expected to merge the dataset config with the ablation config,
    # where the ablation config overrides specific mathematical hyperparameters or architectural flags.
    python train.py \
        --dataset_config "$dataset_config" \
        --ablation_config "$ablation_config" \
        --dataset_root "$DATASET_ROOT" \
        --log_dir "$LOG_ROOT" \
        --checkpoint_dir "$CHECKPOINT_ROOT" \
        --experiment_name "$exp_name" \
        > "$log_file" 2>&1
        
    return $?
}

# =====================================================================
# Main Execution Loop
# =====================================================================

echo "========================================================"
echo "Starting PID-IMVF Ablation Study"
echo "Target Datasets: ${TARGET_DATASETS[*]}"
echo "Ablations: ${ABLATIONS[*]}"
echo "Dataset Root: $DATASET_ROOT"
echo "========================================================"

for dataset in "${TARGET_DATASETS[@]}"; do
    for ablation in "${ABLATIONS[@]}"; do
        
        # Check dependencies before running
        if ! check_dependencies "$dataset" "$ablation"; then
            echo "ERROR: Skipping $ablation on $dataset due to missing configs."
            continue
        fi
        
        # Temporarily disable exit on error to capture failure and continue the loop
        set +e
        run_ablation "$dataset" "$ablation"
        status=$?
        set -e
        
        if [ $status -eq 0 ]; then
            echo "SUCCESS: Completed $ablation on $dataset."
        else
            echo "WARNING: Ablation $ablation on $dataset failed. Check $log_file for details."
        fi
        echo ""
    done
done

echo "========================================================"
echo "Ablation Study Completed."
echo "Logs and PID atom CSVs are available in: $LOG_ROOT"
echo "Model checkpoints are available in: $CHECKPOINT_ROOT"
echo "========================================================"