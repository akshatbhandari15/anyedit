"""
Unified Automated Evaluation Runner for Edited Models
Runs comprehensive evaluation on GSM8K and MMLU benchmarks with a single command.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.evaluate_gsm8k import main as evaluate_gsm8k
from experiments.evaluate_mmlu import main as evaluate_mmlu


def run_automated_evaluation(
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    benchmarks: List[str] = None,
    gsm8k_size: int = None,
    mmlu_size: int = None,
    mmlu_subjects: List[str] = None,
    output_base_dir: str = "output/benchmarks",
    max_new_tokens_gsm8k: int = 512,
    max_new_tokens_mmlu: int = 256,
    temperature: float = 0.001,
    skip_errors: bool = True
):
    """
    Run comprehensive automated evaluation on multiple benchmarks.
    
    Args:
        alg_name: Name of the editing algorithm
        model_name: HuggingFace model name
        hparams_fname: Hyperparameters file name
        benchmarks: List of benchmarks to run ['gsm8k', 'mmlu'] (default: both)
        gsm8k_size: Number of GSM8K samples (None = all)
        mmlu_size: Number of MMLU samples (None = all)
        mmlu_subjects: Specific MMLU subjects (None = all)
        output_base_dir: Base directory for all outputs
        max_new_tokens_gsm8k: Max tokens for GSM8K generation
        max_new_tokens_mmlu: Max tokens for MMLU generation
        temperature: Sampling temperature
        skip_errors: Continue evaluation even if one benchmark fails
    
    Returns:
        Dictionary with results from all benchmarks
    """
    if benchmarks is None:
        benchmarks = ['gsm8k', 'mmlu']
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{alg_name}_{Path(model_name).name}_{timestamp}"
    run_dir = Path(output_base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("AUTOMATED BENCHMARK EVALUATION")
    print("="*80)
    print(f"Run Name: {run_name}")
    print(f"Algorithm: {alg_name}")
    print(f"Model: {model_name}")
    print(f"Hyperparameters: {hparams_fname}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Output Directory: {run_dir}")
    print("="*80)
    
    # Store all results
    all_results = {
        'config': {
            'alg_name': alg_name,
            'model_name': model_name,
            'hparams_fname': hparams_fname,
            'benchmarks': benchmarks,
            'timestamp': timestamp,
            'run_name': run_name
        },
        'benchmarks': {}
    }
    
    # Run GSM8K evaluation
    if 'gsm8k' in benchmarks:
        print("\n" + "="*80)
        print("RUNNING GSM8K EVALUATION")
        print("="*80)
        try:
            gsm8k_output_dir = str(run_dir / "gsm8k")
            gsm8k_metrics, gsm8k_results = evaluate_gsm8k(
                alg_name=alg_name,
                model_name=model_name,
                hparams_fname=hparams_fname,
                dataset_size_limit=gsm8k_size,
                output_dir=gsm8k_output_dir,
                max_new_tokens=max_new_tokens_gsm8k,
                temperature=temperature
            )
            all_results['benchmarks']['gsm8k'] = {
                'status': 'success',
                'metrics': gsm8k_metrics
            }
            print("\n✓ GSM8K evaluation completed successfully")
        except Exception as e:
            error_msg = f"Error in GSM8K evaluation: {str(e)}"
            print(f"\n✗ {error_msg}")
            all_results['benchmarks']['gsm8k'] = {
                'status': 'failed',
                'error': error_msg
            }
            if not skip_errors:
                raise
    
    # Run MMLU evaluation
    if 'mmlu' in benchmarks:
        print("\n" + "="*80)
        print("RUNNING MMLU EVALUATION")
        print("="*80)
        try:
            mmlu_output_dir = str(run_dir / "mmlu")
            mmlu_metrics, mmlu_results = evaluate_mmlu(
                alg_name=alg_name,
                model_name=model_name,
                hparams_fname=hparams_fname,
                dataset_size_limit=mmlu_size,
                subjects=mmlu_subjects,
                output_dir=mmlu_output_dir,
                max_new_tokens=max_new_tokens_mmlu,
                temperature=temperature
            )
            all_results['benchmarks']['mmlu'] = {
                'status': 'success',
                'metrics': mmlu_metrics
            }
            print("\n✓ MMLU evaluation completed successfully")
        except Exception as e:
            error_msg = f"Error in MMLU evaluation: {str(e)}"
            print(f"\n✗ {error_msg}")
            all_results['benchmarks']['mmlu'] = {
                'status': 'failed',
                'error': error_msg
            }
            if not skip_errors:
                raise
    
    # Save summary results
    summary_file = run_dir / "evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for benchmark, result in all_results['benchmarks'].items():
        status = "✓" if result['status'] == 'success' else "✗"
        print(f"\n{status} {benchmark.upper()}: {result['status']}")
        if result['status'] == 'success':
            metrics = result['metrics']
            if 'accuracy' in metrics:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
            if 'correct' in metrics and 'total' in metrics:
                print(f"  Correct: {metrics['correct']}/{metrics['total']}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)
    print(f"Summary saved to: {summary_file}")
    print("="*80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Automated evaluation runner for edited models on GSM8K and MMLU benchmarks"
    )
    
    # Required arguments
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit", "AlphaEdit_ARE", "MEMIT", "MEMIT_ARE", "unke", "unke_ARE"],
        required=True,
        help="Editing algorithm to use"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct)"
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        required=True,
        help="Name of hyperparameters file in hparams/<alg_name>/ folder"
    )
    
    # Benchmark selection
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs='+',
        choices=['gsm8k', 'mmlu', 'both'],
        default=['both'],
        help="Benchmarks to run (default: both)"
    )
    
    # Dataset size limits
    parser.add_argument(
        "--gsm8k_size",
        type=int,
        default=None,
        help="Number of GSM8K samples to evaluate (default: all ~1300)"
    )
    parser.add_argument(
        "--mmlu_size",
        type=int,
        default=None,
        help="Number of MMLU samples to evaluate (default: all ~14000)"
    )
    parser.add_argument(
        "--mmlu_subjects",
        type=str,
        nargs='+',
        default=None,
        help="Specific MMLU subjects to evaluate (default: all 57 subjects)"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/benchmarks",
        help="Base directory for all outputs"
    )
    
    # Generation settings
    parser.add_argument(
        "--max_new_tokens_gsm8k",
        type=int,
        default=512,
        help="Maximum tokens to generate for GSM8K"
    )
    parser.add_argument(
        "--max_new_tokens_mmlu",
        type=int,
        default=256,
        help="Maximum tokens to generate for MMLU"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="Sampling temperature (lower = more deterministic)"
    )
    
    # Error handling
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop evaluation if any benchmark fails (default: continue)"
    )
    
    args = parser.parse_args()
    
    # Process benchmark selection
    benchmarks = args.benchmarks
    if 'both' in benchmarks:
        benchmarks = ['gsm8k', 'mmlu']
    
    # Run evaluation
    results = run_automated_evaluation(
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        benchmarks=benchmarks,
        gsm8k_size=args.gsm8k_size,
        mmlu_size=args.mmlu_size,
        mmlu_subjects=args.mmlu_subjects,
        output_base_dir=args.output_dir,
        max_new_tokens_gsm8k=args.max_new_tokens_gsm8k,
        max_new_tokens_mmlu=args.max_new_tokens_mmlu,
        temperature=args.temperature,
        skip_errors=not args.fail_fast
    )
    
    return results


if __name__ == "__main__":
    main()
