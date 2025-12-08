# Quick Start: Evaluate Edited Models on GSM8K & MMLU

## What This Does

This automated workflow:
1. **Applies edits** to your model using the specified algorithm (MEMIT_ARE, etc.)
2. **Keeps the edits** in memory 
3. **Evaluates** the edited model on GSM8K and MMLU benchmarks
4. **Saves results** with clear metrics

## Quick Start (Gemma3-1B)

```bash
# Make script executable (first time only)
chmod +x run_gemma_benchmarks.sh

# Run the evaluation
./run_gemma_benchmarks.sh
```

This will:
- Apply 10 edits from UnKE dataset to Gemma3-1B-it
- Evaluate on 100 GSM8K samples (math problems)
- Evaluate on 200 MMLU samples (knowledge questions)
- Save results to `output/edited_benchmarks/`

## Custom Configuration

### Method 1: Direct Python Command

```bash
cd AnyEdit

python3 -m experiments.evaluate_edited_model \
    --alg_name=MEMIT_ARE \
    --model_name=google/gemma-3-1b-it \
    --hparams_fname=Gemma3-1B-it.json \
    --edit_size_limit=10 \
    --gsm8k_size=100 \
    --mmlu_size=200
```

### Method 2: Edit the Shell Script

Edit `run_gemma_benchmarks.sh`:
```bash
EDIT_SIZE=10     # Number of edits to apply
GSM8K_SIZE=100   # GSM8K samples (remove line for all ~1300)
MMLU_SIZE=200    # MMLU samples (remove line for all ~14000)
```

## For Different Models

### Qwen 2.5-3B
```bash
python3 -m experiments.evaluate_edited_model \
    --alg_name=MEMIT_ARE \
    --model_name=Qwen/Qwen2.5-3B-Instruct \
    --hparams_fname=Qwen2.5-3B-Instruct.json \
    --edit_size_limit=10 \
    --gsm8k_size=100 \
    --mmlu_size=200
```

### Llama 3.2-3B
```bash
python3 -m experiments.evaluate_edited_model \
    --alg_name=MEMIT_ARE \
    --model_name=meta-llama/Llama-3.2-3B \
    --hparams_fname=Llama3.2-3B.json \
    --edit_size_limit=10 \
    --gsm8k_size=100 \
    --mmlu_size=200
```

## Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--alg_name` | Editing algorithm | Required | `MEMIT_ARE` |
| `--model_name` | HuggingFace model | Required | `google/gemma-3-1b-it` |
| `--hparams_fname` | Config file | Required | `Gemma3-1B-it.json` |
| `--edit_size_limit` | Number of edits | 10 | `20` |
| `--gsm8k_size` | GSM8K samples | All (~1300) | `100` |
| `--mmlu_size` | MMLU samples | All (~14000) | `200` |
| `--skip_gsm8k` | Skip GSM8K | False | `--skip_gsm8k` |
| `--skip_mmlu` | Skip MMLU | False | `--skip_mmlu` |
| `--output_dir` | Output directory | `output/edited_benchmarks` | `my_results` |

## Understanding the Output

### Console Output
```
================================================================================
APPLYING MODEL EDITS
================================================================================
Algorithm: MEMIT_ARE
Edit dataset size: 10
Applying 10 edits to the model...
Edit application took 45.32 seconds
✓ Model edits applied successfully

================================================================================
EVALUATING ON GSM8K
================================================================================
Dataset size: 100
Processing: 100%|████████████████████| 100/100

================================================================================
GSM8K RESULTS
================================================================================
Accuracy: 0.6500 (65/100)
Extraction Rate: 0.9800

================================================================================
EVALUATING ON MMLU
================================================================================
Dataset size: 200
Processing: 100%|████████████████████| 200/200

================================================================================
MMLU RESULTS
================================================================================
Overall Accuracy: 0.5850 (117/200)
Extraction Rate: 0.9950
```

### Output Files

Located in `output/edited_benchmarks/`:

1. **`MEMIT_ARE_Gemma3-1B-it_gsm8k_edited.json`**
   - Detailed GSM8K results
   - Every question, answer, and prediction
   
2. **`MEMIT_ARE_Gemma3-1B-it_mmlu_edited.json`**
   - Detailed MMLU results
   - Per-subject breakdown
   
3. **`MEMIT_ARE_Gemma3-1B-it_benchmark_summary.json`**
   - Combined summary
   - All metrics in one file

### Example Result File Structure
```json
{
  "metrics": {
    "accuracy": 0.65,
    "correct": 65,
    "total": 100,
    "extraction_rate": 0.98
  },
  "results": [
    {
      "id": 0,
      "question": "Janet has 3 apples...",
      "ground_truth": "5",
      "prediction": "The answer is 5"
    }
  ]
}
```

## Workflow Comparison

### Original Workflow (Your Command)
```bash
# Step 1: Apply edits and evaluate on UnKE
python3 -m experiments.evaluate_uns \
    --alg_name=MEMIT_ARE \
    --model_name=google/gemma-3-1b-it \
    --hparams_fname=Gemma3-1B-it.json \
    --ds_name=unke \
    --dataset_size_limit=10 \
    --num_edits=1

# Step 2: Manually analyze results
python -m experiments.summarize_uns --file_path=output/...
```

### New Automated Workflow
```bash
# One command: Apply edits + evaluate on GSM8K & MMLU
python3 -m experiments.evaluate_edited_model \
    --alg_name=MEMIT_ARE \
    --model_name=google/gemma-3-1b-it \
    --hparams_fname=Gemma3-1B-it.json \
    --edit_size_limit=10 \
    --gsm8k_size=100 \
    --mmlu_size=200
```

## Performance Tips

### Quick Testing (2-5 minutes)
```bash
--edit_size_limit=5 --gsm8k_size=50 --mmlu_size=100
```

### Standard Evaluation (15-30 minutes)
```bash
--edit_size_limit=10 --gsm8k_size=100 --mmlu_size=200
```

### Full Evaluation (1-2 hours)
```bash
--edit_size_limit=10
# No size limits - uses all GSM8K (~1300) and MMLU (~14000) samples
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce sample sizes
--gsm8k_size=50 --mmlu_size=100
```

### Module Not Found
```bash
# Make sure you're in the AnyEdit directory
cd AnyEdit
python3 -m experiments.evaluate_edited_model ...
```

### Import Errors
```bash
# Install dependencies
pip install -r ../requirements.txt
```

## Technical Details

### How Edits Are Applied
The script uses the same editing mechanism as `evaluate_uns.py`:
1. Loads the base model
2. Applies edits using the specified algorithm (MEMIT_ARE, etc.)
3. **Keeps the edits in memory** (doesn't revert them)
4. Evaluates the edited model on benchmarks

### Memory Usage
- Model: ~4-8 GB (depends on model size)
- Edits: ~1-2 GB (temporary)
- Evaluation: ~2-4 GB (batched)
- **Total: ~8-14 GB VRAM**

### Supported Algorithms
- `MEMIT` - Mass-Editing Memory in a Transformer
- `MEMIT_ARE` - MEMIT with Autoregressive Editing (AnyEdit)
- `AlphaEdit` - Alpha-based knowledge editing
- `AlphaEdit_ARE` - AlphaEdit with Autoregressive Editing
- `unke` - UnKE baseline
- `unke_ARE` - UnKE with Autoregressive Editing

## Advanced Usage

### Evaluate Only GSM8K
```bash
python3 -m experiments.evaluate_edited_model \
    --alg_name=MEMIT_ARE \
    --model_name=google/gemma-3-1b-it \
    --hparams_fname=Gemma3-1B-it.json \
    --edit_size_limit=10 \
    --skip_mmlu
```

### Evaluate Only MMLU (Specific Subjects)
```bash
python3 -m experiments.evaluate_edited_model \
    --alg_name=MEMIT_ARE \
    --model_name=google/gemma-3-1b-it \
    --hparams_fname=Gemma3-1B-it.json \
    --edit_size_limit=10 \
    --skip_gsm8k \
    --mmlu_subjects abstract_algebra college_physics
```

### Different Edit Datasets
Currently supports `unke` (UnKE benchmark). To add more:
```python
# Edit experiments/evaluate_edited_model.py
if edit_ds_name == "unke":
    edit_dataset = UnKEDataset(...)
elif edit_ds_name == "counterfact":
    edit_dataset = CounterFactDataset(...)
```

## Questions?

See `BENCHMARK_EVALUATION.md` for detailed documentation on:
- Individual benchmark scripts
- Evaluation metrics
- Output formats
- Performance optimization
