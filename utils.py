import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import AddedToken
import numpy as np

def download_model(model_name_or_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    # Add chat template
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.add_tokens([AddedToken("<|im_start|>"),
                          AddedToken("<|im_end|>")])
    tokenizer.save_pretrained(output_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(output_dir)
    
    print(f"Model and tokenizer are saved to {output_dir}")

def test_chat_template():
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    conversation = [{"role": "user", "content": "Hello!"}, 
                    {"role": "assistant", "content": "Hi there! How can I help you today?"},
                    {"role": "user", "content": "I need help with my computer."},
                    {"role": "assistant", "content": "Sure, what's the problem?"}]

    text = tokenizer.apply_chat_template(conversation, tokenize=False)
    ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    print(tokenizer.convert_ids_to_tokens(ids[0]))
    print(text)

def merge_lora(
    base_model_path: str = 'gemma-2b',
    lora_path: str = 'unsloth-lora-checkpoints',
    output_path: str = 'gemma-2b-lora-chat'
    ):
    
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
    )
    lora_model = PeftModel.from_pretrained(base, lora_path)
    model = lora_model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
def quantize_vector(num_bits):
    # A random vector x
    x = np.array([0.2, 0.7, 0.4, 0.9, 0.1])
    # Determine the range of the vector
    x_min = np.min(x)
    x_max = np.max(x)
    # Calculate the step size based on the number of bits
    step_size = (x_max - x_min) / (2 ** num_bits - 1)
    # Quantize the vector
    x_quantized = np.round((x - x_min) / step_size)
    return x_quantized

def dequantize_vector():
    x_quantized = np.array([2, 11, 6, 15, 0])
    # x_quantized = np.array([32, 191, 96, 255, 0])
    x_min, x_max, num_bits = 0.1, 0.9, 8
    # Calculate the step size based on the number of bits
    step_size = (x_max - x_min) / (2 ** num_bits - 1)
    # Dequantize the vector
    x_dequantized = x_quantized * step_size + x_min
    return x_dequantized

if __name__ == "__main__":
    fire.Fire()