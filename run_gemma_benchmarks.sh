#!/bin/bash
# Automated benchmark evaluation for EDITED Gemma3-1B-it model
# This script applies edits to the model, then evaluates on GSM8K and MMLU

echo "================================================================================"
echo "Running GSM8K and MMLU Benchmarks on EDITED Gemma3-1B-it"
echo "================================================================================"
echo ""

# Navigate to AnyEdit directory
cd "$(dirname "$0")/AnyEdit"

# Configuration
ALG_NAME="MEMIT_ARE"
MODEL_NAME="google/gemma-3-1b-it"
HPARAMS_FILE="Gemma3-1B-it.json"
EDIT_SIZE=10     # Number of edits to apply (matching your command)
GSM8K_SIZE=100   # Use 100 samples for quick evaluation, remove for full eval
MMLU_SIZE=200    # Use 200 samples for quick evaluation, remove for full eval
SAVE_MODEL_PATH="../models/gemma-1b-edited-memit"  # Where to save edited model

echo "Configuration:"
echo "  Algorithm: $ALG_NAME"
echo "  Model: $MODEL_NAME"
echo "  Hyperparameters: $HPARAMS_FILE"
echo "  Number of edits: $EDIT_SIZE"
echo "  GSM8K samples: $GSM8K_SIZE"
echo "  MMLU samples: $MMLU_SIZE"
echo "  Save model to: $SAVE_MODEL_PATH"
echo ""
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Load the model"
echo "  2. Apply $EDIT_SIZE edits from UnKE dataset"
echo "  3. Save edited model to: $SAVE_MODEL_PATH"
echo "  4. Evaluate edited model on GSM8K ($GSM8K_SIZE samples)"
echo "  5. Evaluate edited model on MMLU ($MMLU_SIZE samples)"
echo ""
echo "================================================================================"
echo ""

# Run evaluation with edits applied
python3 -m experiments.evaluate_edited_model \
    --alg_name=$ALG_NAME \
    --model_name=$MODEL_NAME \
    --hparams_fname=$HPARAMS_FILE \
    --edit_ds_name=unke \
    --edit_size_limit=$EDIT_SIZE \
    --gsm8k_size=$GSM8K_SIZE \
    --mmlu_size=$MMLU_SIZE \
    --save_model=$SAVE_MODEL_PATH \
    --output_dir=../output/edited_benchmarks

echo ""
echo "================================================================================"
echo "Evaluation completed!"
echo "Results saved to: output/edited_benchmarks/"
echo "================================================================================"
