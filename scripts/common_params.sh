#!/bin/bash
# ==============================================================================
# Common parameters for all hyperparameter search scripts
# ==============================================================================

# Project root (relative to scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default training parameters
DEFAULT_EPOCHS=6
DEFAULT_BATCH_SIZE=2048
DEFAULT_LR=0.001
DEFAULT_SEED=42
DEFAULT_BACKBONE="ViT-B/16"
DEFAULT_ROOT_PATH="f:/shixi/ailab/ood/FA/my_dataset"
DEFAULT_SHOTS=16
DEFAULT_CLASS_NEGATIVES_PATH="f:/shixi/ailab/ood/FA/class_negatives.json"

# Hyperparameter search grids
LEARNING_RATES=(0.001 0.001 0.002)
BATCH_SIZES=(1024 1024)
SEEDS=(42)

# Loss function coefficients grids
LAMBDA_LLM_NEGATIVES=(0.5 0.5 1.0)
LAMBDA_MIXUP=(0.1 0.5 1.0)
MARGIN_VALUES=(0.5 0.5 1.0)

# Selector parameters grid
NUM_SELECT_VALUES=(40 16 32)
