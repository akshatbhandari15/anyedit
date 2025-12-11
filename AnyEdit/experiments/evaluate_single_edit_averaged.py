"""
Evaluate single-edit models on GSM8K/MMLU and average results across multiple edits.
This matches the AnyEdit paper's setup: batch_size=1, multiple independent editing runs.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import sys
from pathlib import Path
from time import time
from typing import Union, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
from copy import deepcopy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsets import UnKEDataset, GSM8KDataset, MMLUDataset
from memit import MEMITHyperParams, apply_memit_to_model
from memit_ARE import MEMITAREHyperParams, apply_memit_ARE_to_model
from AlphaEdit import AlphaEditHyperParams, apply_AlphaEdit_to_model, get_cov
from AlphaEdit_ARE import AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model
from unke import unkeHyperParams, apply_unke_to_model
from unke_ARE import unkeAREHyperParams, apply_unke_ARE_to_model
from util import nethook
from util.globals import *
from experiments.evaluate_gsm8k import evaluate_gsm8k_accuracy, extract_numerical_answer
from experiments.evaluate_mmlu import evaluate_mmlu_accuracy, extract_answer_letter


ALG_DICT = {
    "unke_ARE": (unkeAREHyperParams, apply_unke_ARE_to_model),
    "unke": (unkeHyperParams, apply_unke_to_model),
    "AlphaEdit_ARE": (AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model),
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_ARE": (MEMITAREHyperParams, apply_memit_ARE_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
}


def set_seed(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def apply_single_edit(model, tok, alg_name, hparams, edit_data, ex_datas):
    """
    Apply a single edit to the model (batch_size=1, matching paper setup).
    
    Args:
        model: Model to edit
        tok: Tokenizer
        alg_name: Editing algorithm name
        hparams: Hyperparameters
        edit_data: Single edit datapoint (as a list with 1 element)
        ex_datas: Alpaca context data
    
    Returns:
        model: Edited model (modified in-place)
        weights_copy: Original weights for restoration
    """
    params_class, apply_algo = ALG_DICT[alg_name]
    
    # Prepare arguments
    random_elements = random.sample(ex_datas, 20)
    ex_args = dict(ex_data=random_elements) if any(alg in alg_name for alg in ["unke", "unke_ARE"]) else dict()
    
    # Apply single edit (batch of 1)
    batch = [edit_data]
    weights_copy = apply_algo(model, tok, hparams, batch, **ex_args)
    
    return model, weights_copy


def restore_weights(model, weights_copy):
    """Restore original model weights."""
    with torch.no_grad():
        for k, v in weights_copy.items():
            nethook.get_parameter(model, k)[...] = v.to("cuda")


def evaluate_on_benchmark(model, tok, benchmark_name, dataset, max_new_tokens=512, temperature=0.001):
    """
    Evaluate the model on a specific benchmark dataset.
    """
    predictions = []
    ground_truths = []
    subjects = []
    results = []
    
    for idx, data in enumerate(tqdm(dataset, desc=f"Evaluating {benchmark_name}", leave=False)):
        # Tokenize question
        question = tok([data['question']], return_tensors='pt', padding=True, add_special_tokens=False)
        
        # Generate answer
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=question['input_ids'].to('cuda'),
                attention_mask=question['attention_mask'].to('cuda'),
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
        
        # Decode output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
        ]
        output = tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        predictions.append(output)
        ground_truths.append(data['answer'])
        
        if 'subject' in data:
            subjects.append(data['subject'])
        
        result = {
            'id': idx,
            'question': data.get('raw_question', data['question']),
            'ground_truth': data['answer'],
            'prediction': output
        }
        
        if 'choices' in data:
            result['choices'] = data['choices']
        if 'subject' in data:
            result['subject'] = data['subject']
        
        results.append(result)
    
    return predictions, ground_truths, subjects, results


def main(
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    edit_ds_name: str = "unke",
    num_edits: int = 10,
    gsm8k_size: int = None,
    mmlu_size: int = None,
    mmlu_subjects: list = None,
    output_dir: str = "output/single_edit_averaged_200_gsm8k",
    skip_gsm8k: bool = False,
    skip_mmlu: bool = False,
):
    """
    Main function: Apply single edits independently, evaluate each, average results.
    
    This matches the AnyEdit paper setup:
    - Batch size = 1 (single edit per run)
    - Multiple independent runs
    - Average performance across runs
    """
    set_seed()
    
    print("="*80)
    print("SINGLE-EDIT AVERAGED EVALUATION (Matching AnyEdit Paper)")
    print("="*80)
    print(f"Algorithm: {alg_name}")
    print(f"Model: {model_name}")
    print(f"Hyperparameters: {hparams_fname}")
    print(f"Number of independent edits: {num_edits}")
    print(f"Batch size per edit: 1 (single-instance editing)")
    print("="*80)
    
    # Load hyperparameters
    params_class, _ = ALG_DICT[alg_name]
    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)
    
    # Load model and tokenizer ONCE
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("✓ Model loaded")
    
    # Load edit dataset
    print(f"\nLoading edit dataset ({edit_ds_name})...")
    if edit_ds_name == "unke":
        edit_dataset = UnKEDataset(DATA_DIR, model_name=hparams.model_name, size=num_edits)
    else:
        raise ValueError(f"Unknown edit dataset: {edit_ds_name}")
    print(f"✓ Edit dataset loaded ({len(edit_dataset)} edits available)")
    
    # Load alpaca data for context
    with open(Path(DATA_DIR)/"alpaca_data.json", 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)
    
    # Format alpaca data based on model
    if 'llama' in hparams.model_name.lower():
        ex_datas = [f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{i['instruction']+i['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{i['output']}""" for i in ex_datas]
    elif 'qwen' in hparams.model_name.lower():
        ex_datas = [f"""<|im_start|>user\n{i['instruction']+i['input']}<|im_end|>\n<|im_start|>assistant\n{i['output']}""" for i in ex_datas]
    elif 'gemma' in hparams.model_name.lower():
        ex_datas = [f"""<bos><start_of_turn>user\n{i['instruction']+i['input']}<end_of_turn>\n<start_of_turn>model\n{i['output']}""" for i in ex_datas]
    
    # Load benchmark datasets
    gsm8k_ds = None
    mmlu_ds = None
    
    if not skip_gsm8k:
        print("\nLoading GSM8K dataset...")
        gsm8k_ds = GSM8KDataset(DATA_DIR, model_name=hparams.model_name, size=gsm8k_size)
        print(f"✓ GSM8K dataset loaded ({len(gsm8k_ds)} problems)")
    
    if not skip_mmlu:
        print("\nLoading MMLU dataset...")
        mmlu_ds = MMLUDataset(
            DATA_DIR, 
            model_name=hparams.model_name, 
            size=mmlu_size,
            subjects=mmlu_subjects
        )
        print(f"✓ MMLU dataset loaded ({len(mmlu_ds)} questions)")
    
    # Store results for each edit
    all_edit_results = []
    
    print("\n" + "="*80)
    print(f"RUNNING {num_edits} INDEPENDENT SINGLE-EDIT EVALUATIONS")
    print("="*80)
    
    # For each edit, apply it independently and evaluate
    for edit_idx in range(num_edits):
        print(f"\n{'='*80}")
        print(f"EDIT {edit_idx + 1}/{num_edits}")
        print(f"{'='*80}")
        
        edit_data = edit_dataset[edit_idx]
        print(f"Edit question: {edit_data['question'][:100]}...")
        
        # Apply single edit
        print(f"Applying edit {edit_idx + 1}...")
        start_time = time()
        model, weights_copy = apply_single_edit(model, tok, alg_name, hparams, edit_data, ex_datas)
        edit_time = time() - start_time
        print(f"✓ Edit applied in {edit_time:.2f}s")
        
        edit_result = {
            'edit_idx': edit_idx,
            'edit_data': {
                'question': edit_data['question'],
                'answer': edit_data.get('answer', 'N/A')
            },
            'edit_time': edit_time,
            'benchmarks': {}
        }
        
        # Evaluate on GSM8K
        if not skip_gsm8k:
            print(f"Evaluating on GSM8K...")
            gsm8k_preds, gsm8k_truths, _, gsm8k_results = evaluate_on_benchmark(
                model, tok, "gsm8k", gsm8k_ds, max_new_tokens=512
            )
            
            gsm8k_metrics = evaluate_gsm8k_accuracy(gsm8k_preds, gsm8k_truths)
            
            print(f"GSM8K Accuracy: {gsm8k_metrics['accuracy']:.4f} ({gsm8k_metrics['correct']}/{gsm8k_metrics['total']})")
            
            edit_result['benchmarks']['gsm8k'] = {
                'metrics': gsm8k_metrics,
                'num_samples': len(gsm8k_results)
            }
        
        # Evaluate on MMLU
        if not skip_mmlu:
            print(f"Evaluating on MMLU...")
            mmlu_preds, mmlu_truths, mmlu_subjects_list, mmlu_results = evaluate_on_benchmark(
                model, tok, "mmlu", mmlu_ds, max_new_tokens=256
            )
            
            mmlu_metrics = evaluate_mmlu_accuracy(mmlu_preds, mmlu_truths, mmlu_subjects_list)
            
            print(f"MMLU Accuracy: {mmlu_metrics['accuracy']:.4f} ({mmlu_metrics['correct']}/{mmlu_metrics['total']})")
            
            edit_result['benchmarks']['mmlu'] = {
                'metrics': mmlu_metrics,
                'num_samples': len(mmlu_results)
            }
        
        all_edit_results.append(edit_result)
        
        # Restore original weights for next edit
        print(f"Restoring original model weights...")
        restore_weights(model, weights_copy)
        print(f"✓ Weights restored")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Compute averaged results
    print("\n" + "="*80)
    print("COMPUTING AVERAGED RESULTS")
    print("="*80)
    
    averaged_results = {
        'config': {
            'alg_name': alg_name,
            'model_name': model_name,
            'hparams_fname': hparams_fname,
            'edit_dataset': edit_ds_name,
            'num_independent_edits': num_edits,
            'batch_size_per_edit': 1,
            'paper_matching': True
        },
        'individual_results': all_edit_results,
        'averaged_metrics': {}
    }
    
    # Average GSM8K results
    if not skip_gsm8k:
        gsm8k_accuracies = [r['benchmarks']['gsm8k']['metrics']['accuracy'] for r in all_edit_results]
        gsm8k_correct = [r['benchmarks']['gsm8k']['metrics']['correct'] for r in all_edit_results]
        gsm8k_total = all_edit_results[0]['benchmarks']['gsm8k']['metrics']['total']
        
        averaged_results['averaged_metrics']['gsm8k'] = {
            'mean_accuracy': np.mean(gsm8k_accuracies),
            'std_accuracy': np.std(gsm8k_accuracies),
            'min_accuracy': np.min(gsm8k_accuracies),
            'max_accuracy': np.max(gsm8k_accuracies),
            'mean_correct': np.mean(gsm8k_correct),
            'total_per_run': gsm8k_total,
            'all_accuracies': gsm8k_accuracies
        }
        
        print(f"\nGSM8K Averaged Results:")
        print(f"  Mean Accuracy: {averaged_results['averaged_metrics']['gsm8k']['mean_accuracy']:.4f} ± {averaged_results['averaged_metrics']['gsm8k']['std_accuracy']:.4f}")
        print(f"  Range: [{averaged_results['averaged_metrics']['gsm8k']['min_accuracy']:.4f}, {averaged_results['averaged_metrics']['gsm8k']['max_accuracy']:.4f}]")
        print(f"  Mean Correct: {averaged_results['averaged_metrics']['gsm8k']['mean_correct']:.1f}/{gsm8k_total}")
    
    # Average MMLU results
    if not skip_mmlu:
        mmlu_accuracies = [r['benchmarks']['mmlu']['metrics']['accuracy'] for r in all_edit_results]
        mmlu_correct = [r['benchmarks']['mmlu']['metrics']['correct'] for r in all_edit_results]
        mmlu_total = all_edit_results[0]['benchmarks']['mmlu']['metrics']['total']
        
        averaged_results['averaged_metrics']['mmlu'] = {
            'mean_accuracy': np.mean(mmlu_accuracies),
            'std_accuracy': np.std(mmlu_accuracies),
            'min_accuracy': np.min(mmlu_accuracies),
            'max_accuracy': np.max(mmlu_accuracies),
            'mean_correct': np.mean(mmlu_correct),
            'total_per_run': mmlu_total,
            'all_accuracies': mmlu_accuracies
        }
        
        print(f"\nMMLU Averaged Results:")
        print(f"  Mean Accuracy: {averaged_results['averaged_metrics']['mmlu']['mean_accuracy']:.4f} ± {averaged_results['averaged_metrics']['mmlu']['std_accuracy']:.4f}")
        print(f"  Range: [{averaged_results['averaged_metrics']['mmlu']['min_accuracy']:.4f}, {averaged_results['averaged_metrics']['mmlu']['max_accuracy']:.4f}]")
        print(f"  Mean Correct: {averaged_results['averaged_metrics']['mmlu']['mean_correct']:.1f}/{mmlu_total}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{alg_name}_{hparams.model_name}_single_edit_averaged_{num_edits}runs.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(averaged_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)
    
    return averaged_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Single-edit averaged evaluation matching AnyEdit paper (batch_size=1)"
    )
    
    # Required arguments
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit", "AlphaEdit_ARE", "MEMIT", "MEMIT_ARE", "unke", "unke_ARE"],
        required=True,
        help="Editing algorithm"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--hparams_fname",
        required=True,
        help="Hyperparameters JSON file"
    )
    
    # Edit configuration
    parser.add_argument(
        "--edit_ds_name",
        default="unke",
        help="Dataset to use for edits (default: unke)"
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=10,
        help="Number of independent single-edit runs (default: 10)"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--gsm8k_size",
        type=int,
        default=None,
        help="Number of GSM8K samples per run (default: all)"
    )
    parser.add_argument(
        "--mmlu_size",
        type=int,
        default=None,
        help="Number of MMLU samples per run (default: all)"
    )
    parser.add_argument(
        "--mmlu_subjects",
        nargs='+',
        default=None,
        help="Specific MMLU subjects (default: all)"
    )
    parser.add_argument(
        "--skip_gsm8k",
        action="store_true",
        help="Skip GSM8K evaluation"
    )
    parser.add_argument(
        "--skip_mmlu",
        action="store_true",
        help="Skip MMLU evaluation"
    )
    parser.add_argument(
        "--output_dir",
        default="output/single_edit_averaged",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    main(
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        edit_ds_name=args.edit_ds_name,
        num_edits=args.num_edits,
        gsm8k_size=args.gsm8k_size,
        mmlu_size=args.mmlu_size,
        mmlu_subjects=args.mmlu_subjects,
        output_dir=args.output_dir,
        skip_gsm8k=args.skip_gsm8k,
        skip_mmlu=args.skip_mmlu,
    )
