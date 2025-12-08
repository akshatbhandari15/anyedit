# Automated Benchmark Evaluation for Edited Models

This directory contains evaluation scripts for testing edited language models on **GSM8K** and **MMLU** benchmarks.

## Overview

The evaluation suite consists of:

1. **Dataset Classes** (`dsets/gsm8k.py`, `dsets/mmlu.py`)
   - Load and format GSM8K and MMLU datasets
   - Handle model-specific prompt formatting (Llama, Qwen, Gemma)

2. **Individual Evaluation Scripts**
   - `evaluate_gsm8k.py` - Evaluates mathematical reasoning
   - `evaluate_mmlu.py` - Evaluates knowledge across 57 subjects

3. **Automated Runner** (`run_automated_eval.py`)
   - Run all benchmarks with a single command
   - Comprehensive result tracking and reporting

## Quick Start

### Run All Benchmarks (Automated)

```bash
python3 -m experiments.run_automated_eval \
    --alg_name=MEMIT_ARE \
    --model_name=google/gemma-3-1b-it \
    --hparams_fname=Gemma3-1B-it.json \
    --gsm8k_size=100 \
    --mmlu_size=100
```

### Run Individual Benchmarks

#### GSM8K (Mathematical Reasoning)
```bash
python3 -m experiments.evaluate_gsm8k \
    --alg_name=MEMIT_ARE \
    --model_name=google/gemma-3-1b-it \
    --hparams_fname=Gemma3-1B-it.json \
    --dataset_size_limit=100
```

#### MMLU (Multitask Knowledge)
```bash
python3 -m experiments.evaluate_mmlu \
    --alg_name=MEMIT_ARE \
    --model_name=google/gemma-3-1b-it \
    --hparams_fname=Gemma3-1B-it.json \
    --dataset_size_limit=100
```

## Command Line Arguments

### Common Arguments (All Scripts)

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--alg_name` | Yes | Editing algorithm | - |
| `--model_name` | Yes | HuggingFace model name | - |
| `--hparams_fname` | Yes | Hyperparameters JSON file | - |
| `--temperature` | No | Sampling temperature | 0.001 |

**Supported Algorithms:** `MEMIT`, `MEMIT_ARE`, `AlphaEdit`, `AlphaEdit_ARE`, `unke`, `unke_ARE`

### Automated Runner (`run_automated_eval.py`)

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--benchmarks` | Which benchmarks to run | `both` | `gsm8k mmlu` |
| `--gsm8k_size` | Number of GSM8K samples | All (~1300) | `100` |
| `--mmlu_size` | Number of MMLU samples | All (~14000) | `1000` |
| `--mmlu_subjects` | Specific MMLU subjects | All 57 | `abstract_algebra history` |
| `--output_dir` | Base output directory | `output/benchmarks` | `my_results` |
| `--max_new_tokens_gsm8k` | Max tokens for GSM8K | 512 | 768 |
| `--max_new_tokens_mmlu` | Max tokens for MMLU | 256 | 128 |
| `--fail_fast` | Stop on first error | False | - |

### Individual Scripts

**GSM8K-specific:**
- `--dataset_size_limit`: Number of samples (default: all ~1300)
- `--max_new_tokens`: Maximum tokens to generate (default: 512)
- `--output_dir`: Output directory (default: `output/gsm8k`)

**MMLU-specific:**
- `--dataset_size_limit`: Number of samples (default: all ~14000)
- `--subjects`: Specific subjects to evaluate (default: all 57)
- `--max_new_tokens`: Maximum tokens to generate (default: 256)
- `--output_dir`: Output directory (default: `output/mmlu`)

## Examples

### Full Evaluation (Quick Test)
```bash
# Test with 50 samples per benchmark
python3 -m experiments.run_automated_eval \
    --alg_name=MEMIT_ARE \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B-Instruct.json \
    --gsm8k_size=50 \
    --mmlu_size=50
```

### Full Evaluation (Complete)
```bash
# Run all samples (takes longer)
python3 -m experiments.run_automated_eval \
    --alg_name=MEMIT_ARE \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B-Instruct.json
```

### MMLU Specific Subjects
```bash
# Only evaluate on STEM subjects
python3 -m experiments.evaluate_mmlu \
    --alg_name=MEMIT_ARE \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=Qwen2.5-7B-Instruct.json \
    --subjects abstract_algebra college_chemistry college_physics \
    --dataset_size_limit=200
```

### GSM8K Only (Fast Iteration)
```bash
python3 -m experiments.evaluate_gsm8k \
    --alg_name=MEMIT_ARE \
    --model_name=google/gemma-3-4b-it \
    --hparams_fname=Gemma3-4B-it.json \
    --dataset_size_limit=100 \
    --max_new_tokens=256
```

## Output Structure

### Automated Runner Output
```
output/benchmarks/
└── MEMIT_ARE_gemma-3-1b-it_20241208_143052/
    ├── evaluation_summary.json          # Overall summary
    ├── gsm8k/
    │   └── MEMIT_ARE_Gemma3-1B-it_gsm8k_results.json
    └── mmlu/
        └── MEMIT_ARE_Gemma3-1B-it_mmlu_results.json
```

### Result Files

Each result file contains:
```json
{
  "metrics": {
    "accuracy": 0.75,
    "correct": 75,
    "total": 100,
    "extraction_rate": 0.98
  },
  "results": [
    {
      "id": 0,
      "question": "...",
      "ground_truth": "...",
      "prediction": "..."
    }
  ],
  "config": {
    "alg_name": "MEMIT_ARE",
    "model_name": "...",
    "dataset_size": 100
  }
}
```

### MMLU Per-Subject Results
```json
{
  "metrics": {
    "accuracy": 0.62,
    "subject_accuracy": {
      "abstract_algebra": {"accuracy": 0.70, "correct": 14, "total": 20},
      "college_chemistry": {"accuracy": 0.55, "correct": 11, "total": 20}
    }
  }
}
```

## Benchmarks

### GSM8K (Grade School Math 8K)
- **Task:** Mathematical word problems
- **Size:** ~1,300 problems
- **Metric:** Exact match accuracy
- **Format:** Free-form numerical answer
- **Example:**
  ```
  Q: Janet has 3 apples. She buys 2 more. How many does she have?
  A: 5
  ```

### MMLU (Massive Multitask Language Understanding)
- **Task:** Multiple choice questions across 57 subjects
- **Size:** ~14,000 questions
- **Metric:** Accuracy (A/B/C/D)
- **Categories:** STEM, Humanities, Social Sciences, Other
- **Subjects Include:**
  - STEM: `abstract_algebra`, `college_physics`, `computer_science`
  - Humanities: `philosophy`, `history`, `jurisprudence`
  - Social Sciences: `psychology`, `sociology`, `econometrics`
  - Other: `professional_law`, `professional_medicine`

## Performance Tips

### Memory Optimization
- Use `--dataset_size_limit` to reduce memory usage
- Smaller batch sizes for MMLU subjects: `--mmlu_size=100`

### Speed Optimization
- Reduce `--max_new_tokens` for faster generation
- Run specific subjects: `--mmlu_subjects abstract_algebra`
- Use `--gsm8k_size=100` for quick testing

### Recommended Configurations

**Quick Test (5-10 minutes):**
```bash
--gsm8k_size=50 --mmlu_size=100
```

**Standard Evaluation (30-60 minutes):**
```bash
--gsm8k_size=500 --mmlu_size=1000
```

**Full Evaluation (2-4 hours):**
```bash
# No size limits - evaluates all samples
```

## Troubleshooting

### Dataset Loading Issues
If datasets fail to load:
```bash
# Pre-download datasets
python3 -c "from datasets import load_dataset; load_dataset('gsm8k', 'main'); load_dataset('cais/mmlu', 'all')"
```

### CUDA Out of Memory
- Reduce dataset size
- Lower `max_new_tokens`
- Use gradient checkpointing in model config

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Integration with Existing Code

The evaluation scripts integrate seamlessly with the existing AnyEdit framework:
- Uses same hyperparameter files from `hparams/`
- Compatible with all editing algorithms (MEMIT, AlphaEdit, unke)
- Follows same output format as `evaluate_uns.py`

## Citation

If you use these evaluation scripts, please cite the original benchmarks:

**GSM8K:**
```bibtex
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

**MMLU:**
```bibtex
@article{hendrycks2021measuring,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and others},
  journal={ICLR},
  year={2021}
}
```
