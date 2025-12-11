"""
Qualitative Analysis Script
Generate responses from edited models for various prompts to analyze edit impact.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from pathlib import Path
from time import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

from dsets import UnKEDataset, GSM8KDataset
from memit_ARE import MEMITAREHyperParams, apply_memit_ARE_to_model
from util.globals import *


def set_seed(seed=2024):
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def format_prompt(question, model_name):
    """Format question based on model type."""
    if 'llama' in model_name.lower():
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    elif 'qwen' in model_name.lower():
        return f"""<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""
    elif 'gemma' in model_name.lower():
        return f"""<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"""
    else:
        return question


def generate_response(model, tok, prompt, max_new_tokens=512):
    """Generate a response from the model."""
    inputs = tok(prompt, return_tensors='pt', add_special_tokens=False)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'].to('cuda'),
            attention_mask=inputs['attention_mask'].to('cuda'),
            do_sample=True,
            temperature=0.7,  # Higher temperature for more creative responses
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id
        )
    
    # Decode only the new tokens
    generated_ids = generated_ids[0][inputs['input_ids'].shape[1]:]
    response = tok.decode(generated_ids, skip_special_tokens=True)
    
    return response


def apply_single_edit(model, tok, hparams, edit_data):
    """Apply a single edit to the model."""
    batch = [edit_data]
    
    weights_copy = apply_memit_ARE_to_model(
        model, tok, hparams, batch
    )
    
    return weights_copy


def main(
    model_name: str,
    hparams_fname: str,
    output_dir: str = "output/qualitative_analysis",
):
    set_seed()
    
    print("="*80)
    print("QUALITATIVE ANALYSIS")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Hyperparameters: {hparams_fname}")
    print("="*80)
    
    # Load hyperparameters
    params_path = HPARAMS_DIR / "MEMIT_ARE" / hparams_fname
    hparams = MEMITAREHyperParams.from_json(params_path)
    
    # Load model and tokenizer
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("✓ Model loaded")
    
    # Load datasets
    print("\nLoading datasets...")
    unke_ds = UnKEDataset(DATA_DIR, model_name=hparams.model_name, size=50)
    gsm8k_ds = GSM8KDataset(DATA_DIR, model_name=hparams.model_name, size=50)
    print(f"✓ UnKE dataset: {len(unke_ds)} samples")
    print(f"✓ GSM8K dataset: {len(gsm8k_ds)} samples")
    
    # Load alpaca data for context
    with open(Path(DATA_DIR)/"alpaca_data.json", 'r', encoding='utf-8') as f:
        ex_datas = json.load(f)
    
    # Format alpaca data based on model
    if 'llama' in hparams.model_name.lower():
        ex_datas = [format_prompt(i['instruction']+i['input'], model_name) + i['output'] for i in ex_datas]
    elif 'qwen' in hparams.model_name.lower():
        ex_datas = [format_prompt(i['instruction']+i['input'], model_name) + i['output'] for i in ex_datas]
    elif 'gemma' in hparams.model_name.lower():
        ex_datas = [format_prompt(i['instruction']+i['input'], model_name) + i['output'] for i in ex_datas]
    
    # Select edit sample (first from UnKE)
    edit_sample = unke_ds[0]
    
    # Select other samples
    unke_random_sample = unke_ds[random.randint(1, min(20, len(unke_ds)-1))]
    gsm8k_sample = gsm8k_ds[random.randint(0, min(20, len(gsm8k_ds)-1))]
    
    # Define additional interesting prompts
    additional_prompts = [
        {
            "id": "tiramisu_recipe",
            "question": "Generate me a recipe for Tiramisu",
            "category": "Creative Writing"
        },
        {
            "id": "capital_france",
            "question": "What is the capital of France?",
            "category": "General Knowledge"
        },
        {
            "id": "code_fibonacci",
            "question": "Write a Python function to compute the nth Fibonacci number.",
            "category": "Code Generation"
        },
        {
            "id": "explain_quantum",
            "question": "Explain quantum computing in simple terms.",
            "category": "Technical Explanation"
        },
        {
            "id": "climate_change",
            "question": "What are the main causes of climate change?",
            "category": "Science"
        }
    ]
    
    results = {
        "model_name": model_name,
        "hparams_fname": hparams_fname,
        "edit_info": {
            "question": edit_sample['question'],
            "answer": edit_sample.get('answer', 'N/A'),
            "category": "UnKE (Edit Sample)"
        },
        "before_edit": {},
        "after_edit": {}
    }
    
    # ========================================
    # BEFORE EDIT: Generate all responses
    # ========================================
    print("\n" + "="*80)
    print("GENERATING RESPONSES - BEFORE EDIT")
    print("="*80)
    
    # 1. Edit sample (what we'll edit on)
    print("\n1. Edit Sample (UnKE):")
    print(f"   Question: {edit_sample['question'][:100]}...")
    prompt = format_prompt(edit_sample['question'], model_name)
    response = generate_response(model, tok, prompt)
    print(f"   Response: {response[:150]}...")
    results["before_edit"]["edit_sample"] = {
        "question": edit_sample['question'],
        "ground_truth": edit_sample.get('answer', 'N/A'),
        "response": response,
        "category": "UnKE (Edit Sample)"
    }
    
    # 2. Random UnKE sample
    print("\n2. Random UnKE Sample:")
    print(f"   Question: {unke_random_sample['question'][:100]}...")
    prompt = format_prompt(unke_random_sample['question'], model_name)
    response = generate_response(model, tok, prompt)
    print(f"   Response: {response[:150]}...")
    results["before_edit"]["unke_random"] = {
        "question": unke_random_sample['question'],
        "ground_truth": unke_random_sample.get('answer', 'N/A'),
        "response": response,
        "category": "UnKE (Random)"
    }
    
    # 3. GSM8K sample
    print("\n3. GSM8K Sample:")
    print(f"   Question: {gsm8k_sample['question'][:100]}...")
    prompt = format_prompt(gsm8k_sample['question'], model_name)
    response = generate_response(model, tok, prompt, max_new_tokens=256)
    print(f"   Response: {response[:150]}...")
    results["before_edit"]["gsm8k"] = {
        "question": gsm8k_sample['question'],
        "ground_truth": gsm8k_sample.get('answer', 'N/A'),
        "response": response,
        "category": "GSM8K (Math)"
    }
    
    # 4-8. Additional prompts
    for i, prompt_data in enumerate(additional_prompts, start=4):
        print(f"\n{i}. {prompt_data['category']}:")
        print(f"   Question: {prompt_data['question']}")
        prompt = format_prompt(prompt_data['question'], model_name)
        max_tokens = 512 if prompt_data['id'] == 'tiramisu_recipe' else 256
        response = generate_response(model, tok, prompt, max_new_tokens=max_tokens)
        print(f"   Response: {response[:150]}...")
        results["before_edit"][prompt_data['id']] = {
            "question": prompt_data['question'],
            "response": response,
            "category": prompt_data['category']
        }
    
    # ========================================
    # APPLY EDIT
    # ========================================
    print("\n" + "="*80)
    print("APPLYING SINGLE EDIT")
    print("="*80)
    print(f"Editing on: {edit_sample['question'][:100]}...")
    
    start_time = time()
    weights_copy = apply_single_edit(model, tok, hparams, edit_sample)
    edit_time = time() - start_time
    print(f"✓ Edit applied in {edit_time:.2f}s")
    
    results["edit_time"] = edit_time
    
    # ========================================
    # AFTER EDIT: Generate all responses
    # ========================================
    print("\n" + "="*80)
    print("GENERATING RESPONSES - AFTER EDIT")
    print("="*80)
    
    # 1. Edit sample (should be edited)
    print("\n1. Edit Sample (UnKE) - SHOULD BE AFFECTED:")
    prompt = format_prompt(edit_sample['question'], model_name)
    response = generate_response(model, tok, prompt)
    print(f"   Response: {response[:150]}...")
    results["after_edit"]["edit_sample"] = {
        "question": edit_sample['question'],
        "ground_truth": edit_sample.get('answer', 'N/A'),
        "response": response,
        "category": "UnKE (Edit Sample)"
    }
    
    # 2. Random UnKE sample
    print("\n2. Random UnKE Sample:")
    prompt = format_prompt(unke_random_sample['question'], model_name)
    response = generate_response(model, tok, prompt)
    print(f"   Response: {response[:150]}...")
    results["after_edit"]["unke_random"] = {
        "question": unke_random_sample['question'],
        "ground_truth": unke_random_sample.get('answer', 'N/A'),
        "response": response,
        "category": "UnKE (Random)"
    }
    
    # 3. GSM8K sample
    print("\n3. GSM8K Sample:")
    prompt = format_prompt(gsm8k_sample['question'], model_name)
    response = generate_response(model, tok, prompt, max_new_tokens=256)
    print(f"   Response: {response[:150]}...")
    results["after_edit"]["gsm8k"] = {
        "question": gsm8k_sample['question'],
        "ground_truth": gsm8k_sample.get('answer', 'N/A'),
        "response": response,
        "category": "GSM8K (Math)"
    }
    
    # 4-8. Additional prompts
    for i, prompt_data in enumerate(additional_prompts, start=4):
        print(f"\n{i}. {prompt_data['category']}:")
        prompt = format_prompt(prompt_data['question'], model_name)
        max_tokens = 512 if prompt_data['id'] == 'tiramisu_recipe' else 256
        response = generate_response(model, tok, prompt, max_new_tokens=max_tokens)
        print(f"   Response: {response[:150]}...")
        results["after_edit"][prompt_data['id']] = {
            "question": prompt_data['question'],
            "response": response,
            "category": prompt_data['category']
        }
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"qualitative_{hparams.model_name}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("QUALITATIVE ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Qualitative analysis: Generate responses before/after single edit"
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
    parser.add_argument(
        "--output_dir",
        default="output/qualitative_analysis",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    main(
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        output_dir=args.output_dir,
    )
