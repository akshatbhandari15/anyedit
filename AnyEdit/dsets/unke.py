import json
from pathlib import Path

from util.globals import *

def get_llama_with_answer(que,ans):
  return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ans}<|eot_id|>"""

def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

def get_llama_without_answer_cot(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease provide a multi-hop explanation for the next question: {que}<|eot_id|>"""

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

def get_qwen_without_answer_cot(que):
    return f"""<|im_start|>user\n Please provide a multi-hop explanation for the next question: {que}<|im_end|>\n<|im_start|>assistant\n"""

def get_vicuna_without_answer(que):
    return f"""USER: {que} ASSISTANT:"""

def get_gemma_without_answer(que):
    # Follows Gemma chat format: BOS + user turn + generation prompt.
    return f"""<bos><start_of_turn>user\n{que}<end_of_turn>\n<start_of_turn>model\n"""

def get_list_gemma_without_answer(que, cot):
    # Gemma does not have a separate CoT template here; reuse the standard prompt.
    if cot:
        return [get_gemma_without_answer(line) for line in que]
    return [get_gemma_without_answer(line) for line in que]
def get_list_llama_without_answer(que, cot):
    if cot == False:
        L = [get_llama_without_answer(line) for line in que]
    else:
        L = [get_llama_without_answer_cot(line) for line in que]
    return L
def get_list_qwen_without_answer(que, cot):
    if cot == False:
        L = [get_qwen_without_answer(line) for line in que]
    else:
        L = [get_qwen_without_answer_cot(line) for line in que]
    return L

class UnKEDataset:

    def __init__(self, data_dir: str, model_name: str, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        with open(data_dir/"UnKE"/"final_data_v3.json", 'r', encoding='utf-8') as json_file:
            raw = json.load(json_file)
        for i in raw:
            #i['para_question'] = i['question']
            if model_name == 'Llama3-8B-Instruct':
                i['question'] = get_llama_without_answer(i['question'])
                i['para_question'] = get_llama_without_answer(i['para_question'])
                i['answer'] = i['answer']+'<|eot_id|>'
                i['sub_question'] = get_list_llama_without_answer(i['sub_question'], False)
            elif model_name in ['Qwen2.5-7B-Instruct','Qwen2.5-3B-Instruct']:
                i['question'] = get_qwen_without_answer(i['question'])
                i['para_question'] = get_qwen_without_answer(i['para_question'])
                i['answer'] = i['answer']+'<|im_end|>'
                i['sub_question'] = get_list_qwen_without_answer(i['sub_question'], False)
            elif 'gemma' in model_name.lower():
                i['question'] = get_gemma_without_answer(i['question'])
                i['para_question'] = get_gemma_without_answer(i['para_question'])
                i['answer'] = i['answer'] + '<end_of_turn>'
                i['sub_question'] = get_list_gemma_without_answer(i['sub_question'], False)

        self._data = raw[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
