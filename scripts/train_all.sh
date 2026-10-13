#!/usr/bin/env bash
# =============================================================================
# PID-IMVF : reproducible training across all six original benchmarks.
#
# Usage:
#   bash scripts/train_all.sh
#   DATASETS="hw uci" EPOCHS=50 bash scripts/train_all.sh
#   DATA_ROOT=/home/phd/datasets OUTPUT_ROOT=./outputs bash scripts/train_all.sh
#   DRY_RUN=1 bash scripts/train_all.sh
#
# Every metric is computed on the original data stored under DATA_ROOT.
# No synthetic data is generated and no metric value is hard-coded.
# =============================================================================
set -Eeuo pipefail

# -----------------------------------------------------------------------------
# 1. Resolve repository paths (independent of the caller's working directory)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${REPO_ROOT}/scripts/train_runner.py"

# -----------------------------------------------------------------------------
# 2. Deterministic numerical environment
#    CUBLAS_WORKSPACE_CONFIG is required by torch.use_deterministic_algorithms.
# -----------------------------------------------------------------------------
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# -----------------------------------------------------------------------------
# 3. Experiment configuration (overridable through the environment)
# -----------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/phd/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACCELERATOR="${ACCELERATOR:-auto}"
DEVICES="${DEVICES:-1}"

# PID-IMVF objective weights:
# L = L_CE - lambda_pred*L_PID + lambda_noise*L_VIB
#       + lambda_syn*L_syn + L_id
LAMBDA_PRED="${LAMBDA_PRED:-1.0}"
LAMBDA_NOISE="${LAMBDA_NOISE:-0.1}"
LAMBDA_SYN="${LAMBDA_SYN:-0.1}"
LAMBDA_ID="${LAMBDA_ID:-1.0}"

# Manuscript protocol: View 2 (zero-based index 1) is removed at test time.
MISSING_INDEX="${MISSING_INDEX:-1}"

DATASETS="${DATASETS:-100leaves animals hw mnist noisy_mnist uci}"
DRY_RUN="${DRY_RUN:-0}"

# -----------------------------------------------------------------------------
# 4. Preflight checks
# -----------------------------------------------------------------------------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python interpreter '${PYTHON_BIN}' not found." >&2
    exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
    echo "[ERROR] Missing object-oriented runner: ${RUNNER}" >&2
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] Original dataset root not found: ${DATA_ROOT}" >&2
    echo "        Place the unmodified datasets there before running." >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

# -----------------------------------------------------------------------------
# 5. Run every dataset sequentially (single-device, reproducible)
# -----------------------------------------------------------------------------
MANIFEST="${LOG_ROOT}/train_all_manifest.tsv"
printf "dataset\tstatus\tlog_file\n" > "${MANIFEST}"

FAILED=0
RUN_COUNT=0

echo "=============================================================="
echo " PID-IMVF | original data root : ${DATA_ROOT}"
echo "          | output root        : ${OUTPUT_ROOT}"
echo "          | datasets           : ${DATASETS}"
echo "          | epochs / batch     : ${EPOCHS} / ${BATCH_SIZE}"
echo "          | seed               : ${SEED}"
echo "=============================================================="

for DATASET in ${DATASETS}; do
    RUN_COUNT=$((RUN_COUNT + 1))
    DATASET_LOG="${LOG_ROOT}/${DATASET}.log"

    CMD=(
        "${PYTHON_BIN}" "${RUNNER}"
        --dataset          "${DATASET}"
        --data-root        "${DATA_ROOT}"
        --output-root      "${OUTPUT_ROOT}"
        --log-root         "${LOG_ROOT}"
        --epochs           "${EPOCHS}"
        --batch-size       "${BATCH_SIZE}"
        --lr               "${LEARNING_RATE}"
        --weight-decay     "${WEIGHT_DECAY}"
        --seed             "${SEED}"
        --num-workers      "${NUM_WORKERS}"
        --accelerator      "${ACCELERATOR}"
        --devices          "${DEVICES}"
        --lambda-pred      "${LAMBDA_PRED}"
        --lambda-noise     "${LAMBDA_NOISE}"
        --lambda-syn       "${LAMBDA_SYN}"
        --lambda-id        "${LAMBDA_ID}"
        --missing-index    "${MISSING_INDEX}"
    )

    echo ""
    echo "--------------------------------------------------------------"
    echo "[${RUN_COUNT}] ${DATASET}"
    echo "--------------------------------------------------------------"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY-RUN: %q ' "${CMD[@]}"
        printf '\n'
        printf "%s\tDryRun\t%s\n" "${DATASET}" "${DATASET_LOG}" >> "${MANIFEST}"
        continue
    fi

    # A failure in one dataset does not hide the status of the remaining ones.
    if "${CMD[@]}" 2>&1 | tee "${DATASET_LOG}"; then
        printf "%s\tOK\t%s\n" "${DATASET}" "${DATASET_LOG}" >> "${MANIFEST}"
        echo "[OK] ${DATASET} -> ${OUTPUT_ROOT}/${DATASET}/results.json"
    else
        printf "%s\tFAILED\t%s\n" "${DATASET}" "${DATASET_LOG}" >> "${MANIFEST}"
        echo "[FAILED] ${DATASET} -- see ${DATASET_LOG}" >&2
        FAILED=1
    fi
done

# -----------------------------------------------------------------------------
# 6. Aggregate the real results into manuscript-ready tables
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" != "1" ]]; then
    echo ""
    echo "=============================================================="
    echo " Aggregated results (original data only)"
    echo "=============================================================="
    "${PYTHON_BIN}" "${RUNNER}" \
        --aggregate \
        --output-root "${OUTPUT_ROOT}" \
        | tee "${LOG_ROOT}/aggregated_results.tex"
fi

echo ""
echo "Run manifest: ${MANIFEST}"

if [[ "${FAILED}" -ne 0 ]]; then
    echo "[ERROR] At least one dataset failed. Inspect the manifest." >&2
    exit 1
fi

echo "[DONE] All requested datasets completed."