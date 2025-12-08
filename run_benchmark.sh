#!/bin/bash
# Automated benchmark evaluation script for edited models
# Usage: ./run_benchmark.sh [options]

set -e

# Default values
ALG_NAME=""
MODEL_NAME=""
HPARAMS_FILE=""
BENCHMARKS="both"
GSM8K_SIZE=""
MMLU_SIZE=""
OUTPUT_DIR="output/benchmarks"
TEMPERATURE="0.001"

# Help message
show_help() {
    cat << EOF
Automated Benchmark Evaluation for Edited Models

Usage: ./run_benchmark.sh --alg ALG_NAME --model MODEL_NAME --hparams HPARAMS_FILE [OPTIONS]

Required Arguments:
  --alg ALG_NAME              Editing algorithm (MEMIT_ARE, MEMIT, AlphaEdit, etc.)
  --model MODEL_NAME          HuggingFace model name
  --hparams HPARAMS_FILE      Hyperparameters JSON file name

Optional Arguments:
  --benchmarks BENCHMARKS     Which benchmarks to run: gsm8k, mmlu, or both (default: both)
  --gsm8k-size SIZE          Number of GSM8K samples (default: all ~1300)
  --mmlu-size SIZE           Number of MMLU samples (default: all ~14000)
  --output-dir DIR           Output directory (default: output/benchmarks)
  --temperature TEMP         Sampling temperature (default: 0.001)
  --help                     Show this help message

Examples:
  # Quick test with 100 samples per benchmark
  ./run_benchmark.sh --alg MEMIT_ARE --model google/gemma-3-1b-it \\
      --hparams Gemma3-1B-it.json --gsm8k-size 100 --mmlu-size 100

  # Full evaluation on all samples
  ./run_benchmark.sh --alg MEMIT_ARE --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --hparams Llama3-8B-Instruct.json

  # Only GSM8K evaluation
  ./run_benchmark.sh --alg MEMIT_ARE --model Qwen/Qwen2.5-7B-Instruct \\
      --hparams Qwen2.5-7B-Instruct.json --benchmarks gsm8k

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --alg)
            ALG_NAME="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --hparams)
            HPARAMS_FILE="$2"
            shift 2
            ;;
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --gsm8k-size)
            GSM8K_SIZE="$2"
            shift 2
            ;;
        --mmlu-size)
            MMLU_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$ALG_NAME" ] || [ -z "$MODEL_NAME" ] || [ -z "$HPARAMS_FILE" ]; then
    echo "Error: Missing required arguments"
    echo ""
    show_help
    exit 1
fi

# Build the command
CMD="python3 -m experiments.run_automated_eval"
CMD="$CMD --alg_name=$ALG_NAME"
CMD="$CMD --model_name=$MODEL_NAME"
CMD="$CMD --hparams_fname=$HPARAMS_FILE"
CMD="$CMD --benchmarks $BENCHMARKS"
CMD="$CMD --output_dir=$OUTPUT_DIR"
CMD="$CMD --temperature=$TEMPERATURE"

if [ -n "$GSM8K_SIZE" ]; then
    CMD="$CMD --gsm8k_size=$GSM8K_SIZE"
fi

if [ -n "$MMLU_SIZE" ]; then
    CMD="$CMD --mmlu_size=$MMLU_SIZE"
fi

# Print configuration
echo "================================================================================"
echo "Automated Benchmark Evaluation"
echo "================================================================================"
echo "Algorithm:    $ALG_NAME"
echo "Model:        $MODEL_NAME"
echo "Hyperparams:  $HPARAMS_FILE"
echo "Benchmarks:   $BENCHMARKS"
if [ -n "$GSM8K_SIZE" ]; then
    echo "GSM8K Size:   $GSM8K_SIZE"
fi
if [ -n "$MMLU_SIZE" ]; then
    echo "MMLU Size:    $MMLU_SIZE"
fi
echo "Output Dir:   $OUTPUT_DIR"
echo "Temperature:  $TEMPERATURE"
echo "================================================================================"
echo ""
echo "Running command:"
echo "$CMD"
echo ""
echo "================================================================================"

# Change to AnyEdit directory
cd "$(dirname "$0")/AnyEdit"

# Run the evaluation
eval $CMD

echo ""
echo "================================================================================"
echo "Evaluation completed! Check results in: $OUTPUT_DIR"
echo "================================================================================"
