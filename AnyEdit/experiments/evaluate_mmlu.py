"""
Evaluation script for MMLU (Massive Multitask Language Understanding) benchmark.
Evaluates knowledge and reasoning across 57 diverse subjects.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import re
from pathlib import Path
from time import time
from typing import Union, Tuple
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from dsets import MMLUDataset
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


def extract_answer_letter(text):
    """
    Extract answer letter (A, B, C, or D) from model output.
    Handles various formats like:
    - "The answer is A"
    - "A"
    - "Answer: B"
    - "I choose C"
    """
    text = text.strip()
    
    # Look for explicit "answer is X" pattern
    pattern1 = r'answer\s+is\s+([A-Da-d])'
    match = re.search(pattern1, text.lower())
    if match:
        return match.group(1).upper()
    
    # Look for "Answer: X" pattern
    pattern2 = r'answer\s*:\s*([A-Da-d])'
    match = re.search(pattern2, text.lower())
    if match:
        return match.group(1).upper()
    
    # Look for single letter at the start
    pattern3 = r'^([A-Da-d])[\s\.\)]'
    match = re.search(pattern3, text)
    if match:
        return match.group(1).upper()
    
    # Look for any occurrence of A, B, C, or D (first occurrence)
    pattern4 = r'([A-Da-d])(?=[\s\.\),]|$)'
    match = re.search(pattern4, text)
    if match:
        return match.group(1).upper()
    
    return None


def evaluate_mmlu_accuracy(predictions, ground_truths, subjects=None):
    """
    Calculate accuracy for MMLU predictions.
    
    Args:
        predictions: List of model predictions
        ground_truths: List of correct answers
        subjects: List of subjects for each question (optional, for per-subject metrics)
    
    Returns:
        Dictionary with accuracy metrics
    """
    correct = 0
    total = len(predictions)
    failed_extractions = 0
    
    # Per-subject metrics
    subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'failed': 0})
    
    for idx, (pred, truth) in enumerate(zip(predictions, ground_truths)):
        extracted_pred = extract_answer_letter(pred)
        truth = str(truth).strip().upper()
        
        subject = subjects[idx] if subjects else 'all'
        subject_stats[subject]['total'] += 1
        
        if extracted_pred is None:
            failed_extractions += 1
            subject_stats[subject]['failed'] += 1
            continue
        
        if extracted_pred == truth:
            correct += 1
            subject_stats[subject]['correct'] += 1
    
    accuracy = correct / total if total > 0 else 0
    extraction_rate = (total - failed_extractions) / total if total > 0 else 0
    
    # Calculate per-subject accuracy
    subject_accuracy = {}
    for subject, stats in subject_stats.items():
        if stats['total'] > 0:
            subject_accuracy[subject] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total'],
                'failed': stats['failed']
            }
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'extraction_rate': extraction_rate,
        'failed_extractions': failed_extractions,
        'subject_accuracy': subject_accuracy
    }


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    dataset_size_limit: int,
    subjects: list = None,
    output_dir: str = "output/mmlu",
    max_new_tokens: int = 256,
    temperature: float = 0.001,
):
    """
    Main evaluation function for MMLU.
    
    Args:
        alg_name: Name of the editing algorithm
        model_name: HuggingFace model name or (model, tokenizer) tuple
        hparams_fname: Hyperparameters file name
        dataset_size_limit: Number of samples to evaluate (None = all)
        subjects: List of MMLU subjects to evaluate (None = all)
        output_dir: Directory to save results
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    print("="*80)
    print(f"MMLU Evaluation")
    print(f"Algorithm: {alg_name}")
    print(f"Model: {model_name}")
    if subjects:
        print(f"Subjects: {', '.join(subjects)}")
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
    
    # Load MMLU dataset
    print("Loading MMLU dataset...")
    ds = MMLUDataset(
        DATA_DIR, 
        model_name=hparams.model_name, 
        size=dataset_size_limit,
        subjects=subjects
    )
    print(f"Dataset size: {len(ds)}")
    
    # Prepare for evaluation
    results = []
    predictions = []
    ground_truths = []
    question_subjects = []
    
    print("\nEvaluating on MMLU...")
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
        question_subjects.append(data['subject'])
        
        result = {
            'id': idx,
            'question': data['raw_question'],
            'choices': data['choices'],
            'ground_truth': data['answer'],
            'prediction': output,
            'subject': data['subject']
        }
        results.append(result)
        
        # Print first few examples
        if idx < 5:
            print(f"\n--- Example {idx + 1} ---")
            print(f"Subject: {data['subject']}")
            print(f"Question: {data['raw_question'][:200]}...")
            print(f"Choices: {data['choices']}")
            print(f"Ground Truth: {data['answer']}")
            print(f"Prediction: {output}")
            extracted = extract_answer_letter(output)
            print(f"Extracted Answer: {extracted}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = evaluate_mmlu_accuracy(predictions, ground_truths, question_subjects)
    
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"Answer Extraction Rate: {metrics['extraction_rate']:.4f}")
    print(f"Failed Extractions: {metrics['failed_extractions']}")
    
    # Print per-subject results
    if metrics['subject_accuracy']:
        print("\n" + "="*80)
        print("PER-SUBJECT RESULTS")
        print("="*80)
        subject_results = sorted(metrics['subject_accuracy'].items(), 
                                key=lambda x: x[1]['accuracy'], reverse=True)
        for subject, stats in subject_results:
            if subject != 'all':
                print(f"{subject:40s}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    print("="*80)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / f"{alg_name}_{hparams.model_name}_mmlu_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results,
            'config': {
                'alg_name': alg_name,
                'model_name': model_name,
                'hparams_fname': hparams_fname,
                'dataset_size': len(ds),
                'subjects': subjects,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    
    return metrics, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate edited models on MMLU benchmark")
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
        "--subjects",
        type=str,
        nargs='+',
        default=None,
        help="Specific MMLU subjects to evaluate (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/mmlu",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
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
        subjects=args.subjects,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
