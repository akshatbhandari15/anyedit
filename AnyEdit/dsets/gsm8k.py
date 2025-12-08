import json
from pathlib import Path
from datasets import load_dataset

from util.globals import *


def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""


def get_gemma_without_answer(que):
    return f"""<bos><start_of_turn>user\n{que}<end_of_turn>\n<start_of_turn>model\n"""


class GSM8KDataset:
    """
    Dataset class for GSM8K (Grade School Math 8K) benchmark.
    Tests mathematical reasoning capabilities.
    """

    def __init__(self, data_dir: str, model_name: str, size=None, split="test", *args, **kwargs):
        # Load GSM8K dataset from HuggingFace
        try:
            raw = load_dataset("gsm8k", "main", split=split)
        except Exception as e:
            print(f"Error loading GSM8K dataset: {e}")
            print("Attempting to load from local cache...")
            raw = load_dataset("gsm8k", "main", split=split, cache_dir=data_dir)
        
        # Process the dataset
        processed_data = []
        for item in raw:
            question = item['question']
            answer = item['answer']
            
            # Extract the numerical answer (usually after ####)
            if '####' in answer:
                numerical_answer = answer.split('####')[1].strip()
                reasoning = answer.split('####')[0].strip()
            else:
                numerical_answer = answer.strip()
                reasoning = ""
            
            # Format based on model
            if model_name == 'Llama3-8B-Instruct':
                formatted_question = get_llama_without_answer(question)
                answer_suffix = '<|eot_id|>'
            elif model_name in ['Qwen2.5-7B-Instruct', 'Qwen2.5-3B-Instruct']:
                formatted_question = get_qwen_without_answer(question)
                answer_suffix = '<|im_end|>'
            elif 'gemma' in model_name.lower():
                formatted_question = get_gemma_without_answer(question)
                answer_suffix = '<end_of_turn>'
            else:
                # Default format
                formatted_question = question
                answer_suffix = ''
            
            processed_data.append({
                'question': formatted_question,
                'answer': numerical_answer,
                'reasoning': reasoning,
                'raw_question': question,
                'answer_suffix': answer_suffix
            })
        
        self._data = processed_data[:size] if size else processed_data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
