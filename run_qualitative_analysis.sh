#!/bin/bash
# Qualitative Analysis Script
# Generate before/after edit responses for various prompts

set -e  # Exit on error

cd /home/ab6174_columbia_edu/anyedit/AnyEdit

echo "=========================================="
echo "Qualitative Analysis"
echo "Generating responses before/after edit"
echo "=========================================="
echo ""

# === Llama 3-8B ===
echo "=========================================="
echo "Running Llama 3-8B Instruct"
echo "=========================================="
python3 -m experiments.qualitative_analysis \
  --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname=Llama3-8B-Instruct.json \
  --output_dir=output/qualitative_analysis

echo ""
echo "✓ Llama 3-8B complete"
echo ""

# === Qwen 2.5-7B ===
echo "=========================================="
echo "Running Qwen 2.5-7B Instruct"
echo "=========================================="
python3 -m experiments.qualitative_analysis \
  --model_name=Qwen/Qwen2.5-7B-Instruct \
  --hparams_fname=Qwen2.5-7B-Instruct.json \
  --output_dir=output/qualitative_analysis

echo ""
echo "✓ Qwen 2.5-7B complete"
echo ""

# === Qwen 2.5-3B ===
echo "=========================================="
echo "Running Qwen 2.5-3B Instruct"
echo "=========================================="
python3 -m experiments.qualitative_analysis \
  --model_name=Qwen/Qwen2.5-3B-Instruct \
  --hparams_fname=Qwen2.5-3B-Instruct.json \
  --output_dir=output/qualitative_analysis

echo ""
echo "✓ Qwen 2.5-3B complete"
echo ""

# === Llama 3.2-3B ===
echo "=========================================="
echo "Running Llama 3.2-3B Instruct"
echo "=========================================="
python3 -m experiments.qualitative_analysis \
  --model_name=meta-llama/Llama-3.2-3B-Instruct \
  --hparams_fname=Llama3.2-3B-Instruct.json \
  --output_dir=output/qualitative_analysis

echo ""
echo "✓ Llama 3.2-3B complete"
echo ""

echo "=========================================="
echo "ALL QUALITATIVE ANALYSES COMPLETE"
echo "=========================================="
echo "Results saved to: output/qualitative_analysis/"
echo ""
echo "Summary:"
ls -lh output/qualitative_analysis/*.json
