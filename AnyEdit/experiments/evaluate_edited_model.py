"""
Evaluate edited models on GSM8K and MMLU benchmarks.
This script first applies model edits, then evaluates on both benchmarks with the edits persisted.
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


def save_edited_model(model, tok, save_path, alg_name, edit_info):
    """
    Save the edited model to disk.
    
    Args:
        model: The edited model
        tok: Tokenizer
        save_path: Path to save the model
        alg_name: Name of editing algorithm
        edit_info: Dictionary with edit metadata
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving edited model to: {save_path}")
    
    # Save model and tokenizer
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)
    
    # Save edit metadata
    metadata = {
        'algorithm': alg_name,
        'edit_dataset': edit_info.get('dataset', 'unknown'),
        'num_edits': edit_info.get('num_edits', 0),
        'timestamp': time(),
        'model_name': edit_info.get('model_name', 'unknown')
    }
    
    with open(save_path / 'edit_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved successfully")
    print(f"  - Model weights: {save_path}")
    print(f"  - Tokenizer: {save_path}")
    print(f"  - Metadata: {save_path}/edit_metadata.json")


def apply_model_edits(model, tok, alg_name, hparams, edit_dataset, edit_size_limit=10, save_model_path=None):
    """
    Apply edits to the model using the specified algorithm and dataset.
    Returns the edited model (edits are applied in-place).
    
    Args:
        model: Model to edit
        tok: Tokenizer
        alg_name: Editing algorithm name
        hparams: Hyperparameters
        edit_dataset: Dataset for edits
        edit_size_limit: Number of edits to apply
        save_model_path: Optional path to save edited model
    """
    print("\n" + "="*80)
    print("APPLYING MODEL EDITS")
    print("="*80)
    print(f"Algorithm: {alg_name}")
    print(f"Edit dataset size: {edit_size_limit}")
    
    # Load edit dataset
    ds = edit_dataset
    
    # Load alpaca data for context (if needed)
    with open(Path(DATA_DIR)/"alpaca_data.json", 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)
    
    # Format alpaca data based on model
    if hparams.model_name == 'Llama3-8B-Instruct':
        ex_datas = [f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{i['instruction']+i['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{i['output']}""" for i in ex_datas]
    elif hparams.model_name in ['Qwen2.5-7B-Instruct','Qwen2.5-3B-Instruct']:
        ex_datas = [f"""<|im_start|>user\n{i['instruction']+i['input']}<|im_end|>\n<|im_start|>assistant\n{i['output']}""" for i in ex_datas]
    
    # Prepare projection matrix for AlphaEdit
    P = None
    if any(alg in alg_name for alg in ["AlphaEdit","AlphaEdit_ARE"]):
        if not os.path.exists(f"{hparams.model_name}_null_space_project.pt"):
            print("Computing null space projection matrix...")
            W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
            P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            del W_out
            for i, layer in enumerate(hparams.layers):
                P[i,:,:] = get_project(model, tok, layer, hparams)
            torch.save(P, f"{hparams.model_name}_null_space_project.pt")
        else:
            P = torch.load(f"{hparams.model_name}_null_space_project.pt")
    
    # Apply edits
    params_class, apply_algo = ALG_DICT[alg_name]
    
    batch = ds[:edit_size_limit]
    random_elements = random.sample(ex_datas, 20)
    
    ex_args = dict(ex_data=random_elements) if any(alg in alg_name for alg in ["unke", "unke_ARE"]) else dict()
    nc_args = dict(P=P) if any(alg in alg_name for alg in ["AlphaEdit","AlphaEdit_ARE"]) else dict()
    
    print(f"Applying {len(batch)} edits to the model...")
    start = time()
    weights_copy = apply_algo(model, tok, hparams, batch, **ex_args, **nc_args)
    exec_time = time() - start
    print(f"Edit application took {exec_time:.2f} seconds")
    print("✓ Model edits applied successfully")
    print("="*80)
    
    # Save edited model if requested
    if save_model_path:
        edit_info = {
            'dataset': 'unke',  # Can be parameterized if needed
            'num_edits': len(batch),
            'model_name': hparams.model_name
        }
        save_edited_model(model, tok, save_model_path, alg_name, edit_info)
    
    # Note: weights_copy contains the original weights, but we keep the edited model
    # The model is now edited and ready for benchmark evaluation
    return model, weights_copy


def evaluate_on_benchmark(model, tok, benchmark_name, dataset, max_new_tokens=512, temperature=0.001):
    """
    Evaluate the model on a specific benchmark dataset.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ON {benchmark_name.upper()}")
    print(f"{'='*80}")
    print(f"Dataset size: {len(dataset)}")
    
    predictions = []
    ground_truths = []
    subjects = []
    results = []
    
    for idx, data in enumerate(tqdm(dataset, desc=f"Evaluating {benchmark_name}")):
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
        
        # Print first few examples
        if idx < 3:
            print(f"\n--- Example {idx + 1} ---")
            print(f"Question: {data.get('raw_question', data['question'])[:200]}...")
            print(f"Ground Truth: {data['answer']}")
            print(f"Prediction: {output[:200]}...")
            if benchmark_name == 'gsm8k':
                print(f"Extracted: {extract_numerical_answer(output)}")
            else:
                print(f"Extracted: {extract_answer_letter(output)}")
    
    return predictions, ground_truths, subjects, results


def main(
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    edit_ds_name: str = "unke",
    edit_size_limit: int = 10,
    gsm8k_size: int = None,
    mmlu_size: int = None,
    mmlu_subjects: list = None,
    output_dir: str = "output/edited_benchmarks",
    skip_gsm8k: bool = False,
    skip_mmlu: bool = False,
    save_model_path: str = None,
):
    """
    Main function: Apply edits to model, then evaluate on GSM8K and MMLU.
    """
    set_seed()
    
    print("="*80)
    print("AUTOMATED EVALUATION OF EDITED MODEL")
    print("="*80)
    print(f"Algorithm: {alg_name}")
    print(f"Model: {model_name}")
    print(f"Hyperparameters: {hparams_fname}")
    print(f"Edit dataset: {edit_ds_name} (size: {edit_size_limit})")
    print("="*80)
    
    # Load hyperparameters
    params_class, _ = ALG_DICT[alg_name]
    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)
    
    # Load model and tokenizer
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("✓ Model loaded")
    
    # Load edit dataset
    print(f"\nLoading edit dataset ({edit_ds_name})...")
    if edit_ds_name == "unke":
        edit_dataset = UnKEDataset(DATA_DIR, model_name=hparams.model_name, size=edit_size_limit)
    else:
        raise ValueError(f"Unknown edit dataset: {edit_ds_name}")
    print("✓ Edit dataset loaded")
    
    # Apply edits to the model
    model, original_weights = apply_model_edits(
        model, tok, alg_name, hparams, edit_dataset, edit_size_limit,
        save_model_path=save_model_path
    )
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'config': {
            'alg_name': alg_name,
            'model_name': model_name,
            'hparams_fname': hparams_fname,
            'edit_dataset': edit_ds_name,
            'edit_size': edit_size_limit,
        },
        'benchmarks': {}
    }
    
    # Evaluate on GSM8K
    if not skip_gsm8k:
        print("\nLoading GSM8K dataset...")
        gsm8k_ds = GSM8KDataset(DATA_DIR, model_name=hparams.model_name, size=gsm8k_size)
        print("✓ GSM8K dataset loaded")
        
        gsm8k_preds, gsm8k_truths, _, gsm8k_results = evaluate_on_benchmark(
            model, tok, "gsm8k", gsm8k_ds, max_new_tokens=512
        )
        
        gsm8k_metrics = evaluate_gsm8k_accuracy(gsm8k_preds, gsm8k_truths)
        
        print("\n" + "="*80)
        print("GSM8K RESULTS")
        print("="*80)
        print(f"Accuracy: {gsm8k_metrics['accuracy']:.4f} ({gsm8k_metrics['correct']}/{gsm8k_metrics['total']})")
        print(f"Extraction Rate: {gsm8k_metrics['extraction_rate']:.4f}")
        print("="*80)
        
        all_results['benchmarks']['gsm8k'] = {
            'metrics': gsm8k_metrics,
            'results': gsm8k_results
        }
        
        # Save GSM8K results
        gsm8k_file = output_path / f"{alg_name}_{hparams.model_name}_gsm8k_edited.json"
        with open(gsm8k_file, 'w', encoding='utf-8') as f:
            json.dump(all_results['benchmarks']['gsm8k'], f, indent=2, ensure_ascii=False)
        print(f"GSM8K results saved to: {gsm8k_file}")
    
    # Evaluate on MMLU
    if not skip_mmlu:
        print("\nLoading MMLU dataset...")
        mmlu_ds = MMLUDataset(
            DATA_DIR, 
            model_name=hparams.model_name, 
            size=mmlu_size,
            subjects=mmlu_subjects
        )
        print("✓ MMLU dataset loaded")
        
        mmlu_preds, mmlu_truths, mmlu_subjects_list, mmlu_results = evaluate_on_benchmark(
            model, tok, "mmlu", mmlu_ds, max_new_tokens=256
        )
        
        mmlu_metrics = evaluate_mmlu_accuracy(mmlu_preds, mmlu_truths, mmlu_subjects_list)
        
        print("\n" + "="*80)
        print("MMLU RESULTS")
        print("="*80)
        print(f"Overall Accuracy: {mmlu_metrics['accuracy']:.4f} ({mmlu_metrics['correct']}/{mmlu_metrics['total']})")
        print(f"Extraction Rate: {mmlu_metrics['extraction_rate']:.4f}")
        print("="*80)
        
        all_results['benchmarks']['mmlu'] = {
            'metrics': mmlu_metrics,
            'results': mmlu_results
        }
        
        # Save MMLU results
        mmlu_file = output_path / f"{alg_name}_{hparams.model_name}_mmlu_edited.json"
        with open(mmlu_file, 'w', encoding='utf-8') as f:
            json.dump(all_results['benchmarks']['mmlu'], f, indent=2, ensure_ascii=False)
        print(f"MMLU results saved to: {mmlu_file}")
    
    # Save combined summary
    summary_file = output_path / f"{alg_name}_{hparams.model_name}_benchmark_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Summary saved to: {summary_file}")
    print("="*80)
    
    return all_results


def get_project(model, tok, layer, hparams):
    """Compute null space projection for AlphaEdit"""
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(f"Found {len(small_singular_indices)} small singular values")
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply edits to a model and evaluate on GSM8K and MMLU benchmarks"
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
        "--edit_size_limit",
        type=int,
        default=10,
        help="Number of edits to apply (default: 10)"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--gsm8k_size",
        type=int,
        default=None,
        help="Number of GSM8K samples (default: all)"
    )
    parser.add_argument(
        "--mmlu_size",
        type=int,
        default=None,
        help="Number of MMLU samples (default: all)"
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
    parser.add_argument(
        "--output_dir",
        default="output/edited_benchmarks",
        help="Output directory"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
    main(
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        edit_ds_name=args.edit_ds_name,
        edit_size_limit=args.edit_size_limit,
        gsm8k_size=args.gsm8k_size,
        mmlu_size=args.mmlu_size,
        mmlu_subjects=args.mmlu_subjects,
        output_dir=args.output_dir,
        skip_gsm8k=args.skip_gsm8k,
        skip_mmlu=args.skip_mmlu,
        save_model_path=args.save_model,
    )   gsm8k_size=args.gsm8k_size,
        mmlu_size=args.mmlu_size,
        mmlu_subjects=args.mmlu_subjects,
        output_dir=args.output_dir,
        skip_gsm8k=args.skip_gsm8k,
        skip_mmlu=args.skip_mmlu,
    )
