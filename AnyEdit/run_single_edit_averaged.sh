#!/bin/bash
# Single-Edit Averaged Evaluation Script
# Matches AnyEdit paper methodology (batch_size=1)
# Runs 10 independent single-edit runs and averages results

set -e  # Exit on error

cd /home/ab6174_columbia_edu/anyedit/AnyEdit

echo "=========================================="
echo "Single-Edit Averaged Evaluation"
echo "Paper-matching methodology: batch_size=1"
echo "=========================================="
echo ""

# Configuration
ALG_NAME="MEMIT_ARE"
NUM_EDITS=10
GSM8K_SIZE=200
EDIT_DS_NAME="unke"

# === Llama 3-8B ===
echo "=========================================="
echo "Running Llama 3-8B Instruct"
echo "=========================================="
python3 -m experiments.evaluate_single_edit_averaged \
  --alg_name=${ALG_NAME} \
  --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname=Llama3-8B-Instruct.json \
  --edit_ds_name=${EDIT_DS_NAME} \
  --num_edits=${NUM_EDITS} \
  --gsm8k_size=${GSM8K_SIZE} \
  --skip_mmlu

echo ""
echo "✓ Llama 3-8B complete"
echo ""

# === Qwen 2.5-7B ===
echo "=========================================="
echo "Running Qwen 2.5-7B Instruct"
echo "=========================================="
python3 -m experiments.evaluate_single_edit_averaged \
  --alg_name=${ALG_NAME} \
  --model_name=Qwen/Qwen2.5-7B-Instruct \
  --hparams_fname=Qwen2.5-7B-Instruct.json \
  --edit_ds_name=${EDIT_DS_NAME} \
  --num_edits=${NUM_EDITS} \
  --gsm8k_size=${GSM8K_SIZE} \
  --skip_mmlu

echo ""
echo "✓ Qwen 2.5-7B complete"
echo ""

# === Qwen 2.5-3B ===
echo "=========================================="
echo "Running Qwen 2.5-3B Instruct"
echo "=========================================="
python3 -m experiments.evaluate_single_edit_averaged \
  --alg_name=${ALG_NAME} \
  --model_name=Qwen/Qwen2.5-3B-Instruct \
  --hparams_fname=Qwen2.5-3B-Instruct.json \
  --edit_ds_name=${EDIT_DS_NAME} \
  --num_edits=${NUM_EDITS} \
  --gsm8k_size=${GSM8K_SIZE} \
  --skip_mmlu

echo ""
echo "✓ Qwen 2.5-3B complete"
echo ""

# === Llama 3.2-3B ===
echo "=========================================="
echo "Running Llama 3.2-3B Instruct"
echo "=========================================="
python3 -m experiments.evaluate_single_edit_averaged \
  --alg_name=${ALG_NAME} \
  --model_name=meta-llama/Llama-3.2-3B-Instruct \
  --hparams_fname=Llama3.2-3B-Instruct.json \
  --edit_ds_name=${EDIT_DS_NAME} \
  --num_edits=${NUM_EDITS} \
  --gsm8k_size=${GSM8K_SIZE} \
  --skip_mmlu

echo ""
echo "✓ Llama 3.2-3B complete"
echo ""

# === Gemma 3-1B ===
echo "=========================================="
echo "Running Gemma 3-1B IT"
echo "=========================================="
python3 -m experiments.evaluate_single_edit_averaged \
  --alg_name=${ALG_NAME} \
  --model_name=google/gemma-3-1b-it \
  --hparams_fname=Gemma3-1B-it.json \
  --edit_ds_name=${EDIT_DS_NAME} \
  --num_edits=${NUM_EDITS} \
  --gsm8k_size=${GSM8K_SIZE} \
  --skip_mmlu

echo ""
echo "✓ Gemma 3-1B complete"
echo ""

# === Gemma 3-4B ===
echo "=========================================="
echo "Running Gemma 3-4B IT"
echo "=========================================="
python3 -m experiments.evaluate_single_edit_averaged \
  --alg_name=${ALG_NAME} \
  --model_name=google/gemma-3-4b-it \
  --hparams_fname=Gemma3-4B-it.json \
  --edit_ds_name=${EDIT_DS_NAME} \
  --num_edits=${NUM_EDITS} \
  --gsm8k_size=${GSM8K_SIZE} \
  --skip_mmlu

echo ""
echo "✓ Gemma 3-4B complete"
echo ""

echo "=========================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=========================================="
echo "Results saved to: output/single_edit_averaged/"
echo ""
echo "Summary:"
ls -lh output/single_edit_averaged/*.json
