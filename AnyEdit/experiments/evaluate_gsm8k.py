"""
Evaluation script for GSM8K (Grade School Math 8K) benchmark.
Evaluates mathematical reasoning capabilities of edited models.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import re
from pathlib import Path
from time import time
from typing import Union, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from dsets import GSM8KDataset
from memit import MEMITHyperParams, apply_memit_to_model
from memit_ARE import MEMITAREHyperParams, apply_memit_ARE_to_model
from AlphaEdit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from AlphaEdit_ARE import AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model
from unke import unkeHyperParams, apply_unke_to_model
from unke_ARE import unkeAREHyperParams, apply_unke_ARE_to_model
from util.globals import *


ALG_DICT = {
    "unke_ARE": (unkeAREHyperParams, apply_unke_ARE_to_model),
    "unke": (unkeHyperParams, apply_unke_to_model),
    "AlphaEdit_ARE": (AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model),
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_ARE": (MEMITAREHyperParams, apply_memit_ARE_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
}


def extract_numerical_answer(text):
    """
    Extract numerical answer from model output.
    Handles various formats like:
    - "The answer is 42"
    - "42"
    - "#### 42"
    - "So the final answer is 42."
    """
    text = text.strip()
    
    # Look for patterns like "#### number"
    pattern1 = r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern1, text)
    if match:
        return match.group(1).replace(',', '')
    
    # Look for "answer is" pattern
    pattern2 = r'answer\s+is\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern2, text.lower())
    if match:
        return match.group(1).replace(',', '')
    
    # Look for numbers at the end of the text
    pattern3 = r'(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*\.?\s*$'
    match = re.search(pattern3, text)
    if match:
        return match.group(1).replace(',', '')
    
    # Look for any number in the text (last occurrence)
    pattern4 = r'(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    matches = re.findall(pattern4, text)
    if matches:
        return matches[-1].replace(',', '')
    
    return None


def evaluate_gsm8k_accuracy(predictions, ground_truths):
    """
    Calculate accuracy for GSM8K predictions.
    
    Args:
        predictions: List of model predictions
        ground_truths: List of correct answers
    
    Returns:
        Dictionary with accuracy metrics
    """
    correct = 0
    total = len(predictions)
    failed_extractions = 0
    
    for pred, truth in zip(predictions, ground_truths):
        extracted_pred = extract_numerical_answer(pred)
        truth = str(truth).strip().replace(',', '')
        
        if extracted_pred is None:
            failed_extractions += 1
            continue
        
        # Normalize for comparison
        try:
            pred_num = float(extracted_pred)
            truth_num = float(truth)
            if abs(pred_num - truth_num) < 1e-5:  # Allow small floating point differences
                correct += 1
        except ValueError:
            # If conversion fails, do string comparison
            if extracted_pred == truth:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    extraction_rate = (total - failed_extractions) / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'extraction_rate': extraction_rate,
        'failed_extractions': failed_extractions
    }


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    dataset_size_limit: int,
    output_dir: str = "output/gsm8k",
    max_new_tokens: int = 512,
    temperature: float = 0.001,
):
    """
    Main evaluation function for GSM8K.
    
    Args:
        alg_name: Name of the editing algorithm
        model_name: HuggingFace model name or (model, tokenizer) tuple
        hparams_fname: Hyperparameters file name
        dataset_size_limit: Number of samples to evaluate (None = all)
        output_dir: Directory to save results
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    print("="*80)
    print(f"GSM8K Evaluation")
    print(f"Algorithm: {alg_name}")
    print(f"Model: {model_name}")
    print("="*80)
    
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]
    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)
    
    # Load model and tokenizer
    if isinstance(model_name, str):
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    
    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    ds = GSM8KDataset(DATA_DIR, model_name=hparams.model_name, size=dataset_size_limit)
    print(f"Dataset size: {len(ds)}")
    
    # Prepare for evaluation
    results = []
    predictions = []
    ground_truths = []
    
    print("\nEvaluating on GSM8K...")
    for idx, data in enumerate(tqdm(ds, desc="Processing")):
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
        
        # Decode generated output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
        ]
        output = tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Store results
        predictions.append(output)
        ground_truths.append(data['answer'])
        
        result = {
            'id': idx,
            'question': data['raw_question'],
            'ground_truth': data['answer'],
            'prediction': output,
            'reasoning': data['reasoning']
        }
        results.append(result)
        
        # Print first few examples
        if idx < 5:
            print(f"\n--- Example {idx + 1} ---")
            print(f"Question: {data['raw_question']}")
            print(f"Ground Truth: {data['answer']}")
            print(f"Prediction: {output}")
            extracted = extract_numerical_answer(output)
            print(f"Extracted Answer: {extracted}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = evaluate_gsm8k_accuracy(predictions, ground_truths)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"Answer Extraction Rate: {metrics['extraction_rate']:.4f}")
    print(f"Failed Extractions: {metrics['failed_extractions']}")
    print("="*80)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / f"{alg_name}_{hparams.model_name}_gsm8k_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results,
            'config': {
                'alg_name': alg_name,
                'model_name': model_name,
                'hparams_fname': hparams_fname,
                'dataset_size': len(ds),
                'max_new_tokens': max_new_tokens,
                'temperature': temperature
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    
    return metrics, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate edited models on GSM8K benchmark")
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit", "AlphaEdit_ARE", "MEMIT", "MEMIT_ARE", "unke", "unke_ARE"],
        required=True,
        help="Editing algorithm to use"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        required=True,
        help="Name of hyperparameters file in hparams/<alg_name>/ folder"
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/gsm8k",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    main(
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        dataset_size_limit=args.dataset_size_limit,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
