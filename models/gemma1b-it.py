from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch

model_id = "google/gemma-3-1b-it"

# It is best practice to load the MODEL in bfloat16, not the inputs
model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).eval().to("cuda") # Ensure model is on GPU

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is John Mayne's background and experience in journalism?"},]
        },
    ],
]

# ✅ CORRECTED BLOCK
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device) # Only move to device. DO NOT cast to dtype here.

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64)

# Decode only the newly generated tokens
# (outputs includes input tokens, so we usually strip them for cleaner printing)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))