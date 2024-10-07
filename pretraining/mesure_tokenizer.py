from transformers import  AutoTokenizer
import re
import os
from datasets import load_dataset
import numpy as np 



def compute_compression_ratio(text):
    text = keep_text_only(text)
    encoded = tokenizer.tokenize(text, add_special_tokens=False)
    raw_size = len(text.encode('utf-8'))
    tokenized_size = sum(len(token.encode('utf-8')) for token in encoded)
    compression_rate = raw_size / tokenized_size
    return compression_rate

def compute_tokens_per_bype(text):
    tokens = tokenizer.tokenize(text, add_special_tokens=False)
    tokens_per_byte = len(tokens) / len(text.encode('utf-8'))
    return tokens_per_byte

def keep_text_only(text):
    cleaned_text = re.sub(r'\d+|[^\w\s]', '', text)
    return cleaned_text

def measure_tok_per_bype(text):
    num_tokens = sum(tokenizer(text, return_length=True, return_attention_mask=False, add_special_tokens=False, return_token_type_ids=False)["length"])
    num_bytes = sum([len(s.encode('utf-8')) for s in text])
    return {"tokens_per_byte": num_tokens / num_bytes}


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    print("The number of vocab: ", len(tokenizer))
    dataset = load_dataset('TranCongThien/viet_wiki_data', split='train')
    docs = dataset.remove_columns(['meta'])
    docs = docs['text'][:10000]
    print(measure_tok_per_bype(docs)) 