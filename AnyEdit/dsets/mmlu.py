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


def format_question_with_choices(question, choices):
    """Format multiple choice question with A, B, C, D options"""
    formatted = f"{question}\n\n"
    choice_labels = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choices):
        if i < len(choice_labels):
            formatted += f"{choice_labels[i]}. {choice}\n"
    formatted += "\nAnswer:"
    return formatted


class MMLUDataset:
    """
    Dataset class for MMLU (Massive Multitask Language Understanding) benchmark.
    Tests knowledge across 57 subjects including STEM, humanities, social sciences, and more.
    """

    def __init__(self, data_dir: str, model_name: str, size=None, subjects=None, split="test", *args, **kwargs):
        """
        Args:
            data_dir: Directory for caching datasets
            model_name: Name of the model (for prompt formatting)
            size: Maximum number of samples to use (None = all)
            subjects: List of MMLU subjects to include (None = all subjects)
            split: Dataset split to use ('test', 'validation', 'dev')
        """
        # Load MMLU dataset from HuggingFace
        try:
            if subjects is None:
                # Load all subjects
                raw = load_dataset("cais/mmlu", "all", split=split)
            else:
                # Load specific subjects
                all_data = []
                for subject in subjects:
                    subject_data = load_dataset("cais/mmlu", subject, split=split)
                    all_data.extend(subject_data)
                raw = all_data
        except Exception as e:
            print(f"Error loading MMLU dataset: {e}")
            print("Attempting to load from local cache...")
            try:
                raw = load_dataset("cais/mmlu", "all", split=split, cache_dir=data_dir)
            except:
                # Fallback to alternative MMLU dataset
                raw = load_dataset("lukaemon/mmlu", split=split, cache_dir=data_dir)
        
        # Process the dataset
        processed_data = []
        for item in raw:
            # MMLU format: question, choices (A, B, C, D), answer (0-3)
            if isinstance(item, dict):
                question = item.get('question', '')
                choices = item.get('choices', [])
                answer_idx = item.get('answer', 0)
                subject = item.get('subject', 'unknown')
            else:
                # Handle different dataset formats
                question = item['question'] if hasattr(item, 'question') else ''
                choices = item['choices'] if hasattr(item, 'choices') else []
                answer_idx = item['answer'] if hasattr(item, 'answer') else 0
                subject = item['subject'] if hasattr(item, 'subject') else 'unknown'
            
            # Format the question with choices
            formatted_text = format_question_with_choices(question, choices)
            
            # Format based on model
            if model_name == 'Llama3-8B-Instruct':
                formatted_question = get_llama_without_answer(formatted_text)
                answer_suffix = '<|eot_id|>'
            elif model_name in ['Qwen2.5-7B-Instruct', 'Qwen2.5-3B-Instruct']:
                formatted_question = get_qwen_without_answer(formatted_text)
                answer_suffix = '<|im_end|>'
            elif 'gemma' in model_name.lower():
                formatted_question = get_gemma_without_answer(formatted_text)
                answer_suffix = '<end_of_turn>'
            else:
                # Default format
                formatted_question = formatted_text
                answer_suffix = ''
            
            # Convert answer index to letter
            choice_labels = ['A', 'B', 'C', 'D']
            answer_letter = choice_labels[answer_idx] if answer_idx < len(choice_labels) else 'A'
            
            processed_data.append({
                'question': formatted_question,
                'raw_question': question,
                'choices': choices,
                'answer': answer_letter,
                'answer_idx': answer_idx,
                'subject': subject,
                'answer_suffix': answer_suffix
            })
        
        self._data = processed_data[:size] if size else processed_data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
