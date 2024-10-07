import argparse
import os
import fire
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import AutoTokenizer, GemmaTokenizer, GemmaForCausalLM
import torch
import sentencepiece as spm
from datasets import load_dataset

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import json

import re

def has_non_alphabetic_chars(token):
    # Function to check if a token contains non-alphabetic characters
    return any(not char.isalpha() for char in token)

def merge_tokenizer(
    source_tokenizer_dir,
    new_tokenizer_model,
    new_tokenizer_dir):
    
    # Load the source tokenizer
    source_tokenizer = GemmaTokenizer.from_pretrained(source_tokenizer_dir)
    source_sp_processor = source_tokenizer.sp_model
    source_spm = sp_pb2_model.ModelProto()
    source_spm.ParseFromString(source_sp_processor.serialized_model_proto())
    
    source_spm_tokens = set([p.piece for p in source_spm.pieces])
    
    # Load the new tokenizer model
    sp_tgt = spm.SentencePieceProcessor()
    if not new_tokenizer_model.endswith(".model"):
        new_tokenizer_model = new_tokenizer_model + ".model"
    sp_tgt.load(new_tokenizer_model)
    
    sp_tgt_pb2 = sp_pb2_model.ModelProto()
    sp_tgt_pb2.ParseFromString(sp_tgt.serialized_model_proto())
    new_tgt_tokens = list(set([p.piece for p in sp_tgt_pb2.pieces]))
    print("The number of original tokens:", len(source_spm_tokens))
    print("The number of new tokens:", len(new_tgt_tokens))
    
    # Merge the new tokens into the source tokenizer
    for piece in new_tgt_tokens:
        assert isinstance(piece, str), f"Invalid token({piece}) type {type(piece)}"
        if piece in source_spm_tokens:
            # Skip existed token.
            continue
        else:
            # Skip non-alphabetic token.
            if not has_non_alphabetic_chars(piece.replace("▁", "")):
                new_p = sp_pb2_model.ModelProto().SentencePiece()
                new_p.piece = piece
                new_p.score = 0
                source_spm.pieces.append(new_p)
            else:
                print(f"Skip non-alphabetic token {piece}")
        
    print(f"Expand vocab from {len(source_spm_tokens)} to {len(source_spm.pieces)}")
    
    # Save the expanded tokenizer
    os.makedirs(new_tokenizer_dir)
    target_tokenizer_model_path = os.path.join(new_tokenizer_dir, "tokenizer.model")
    with open(file=target_tokenizer_model_path, mode="wb") as fp:
        fp.write(source_spm.SerializeToString())
    
    target_tokenizer = GemmaTokenizer(vocab_file=target_tokenizer_model_path)
    target_tokenizer.save_pretrained(save_directory=new_tokenizer_dir)
    
    
def reinit_model(model_name, new_tokenizer_dir):
    # Load the source tokenizer and model
    source_tokenizer = GemmaTokenizer.from_pretrained(model_name)
    source_tokenizer.add_bos_token = False
    source_tokenizer.add_eos_token = False
    if source_tokenizer.pad_token is None:
        source_tokenizer.pad_token = source_tokenizer.unk_token
    source_vocab = source_tokenizer.get_vocab()
    
    # Load the target tokenizer
    target_tokenizer = GemmaTokenizer.from_pretrained(new_tokenizer_dir)
    target_tokenizer.add_bos_token = False
    target_tokenizer.add_eos_token = False
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.unk_token
    target_vocab = target_tokenizer.get_vocab()
    target_inverted_vocab = {v: k for k, v in target_vocab.items()}
    
    assert len(target_vocab) > len(
        source_vocab
    ), f"Target vocab size({len(target_vocab)}) must be greater than source vocab size({len(source_vocab)})"
    
    gpu_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")
    
    source_model = GemmaForCausalLM.from_pretrained(model_name)
    source_model.eval()
    source_model = source_model.to(gpu_device)
    
    source_input_embeddings = source_model.get_input_embeddings()
    assert isinstance(source_input_embeddings, torch.nn.Embedding)
    assert source_input_embeddings.weight.shape[0] == len(source_vocab)
    source_input_embeddings.eval()
    
    source_output_embeddings = source_model.get_output_embeddings()
    assert isinstance(source_output_embeddings, torch.nn.Linear)
    assert source_output_embeddings.bias is None
    assert source_output_embeddings.weight.shape[0] == len(source_vocab)
    source_output_embeddings.eval()
    
    input_embeddings = source_input_embeddings.weight.cpu().detach().numpy()
    output_embeddings = source_output_embeddings.weight.cpu().detach().numpy()
    
    # Expand the model with new tokens
    for i in range(len(source_vocab), len(target_vocab)):
        if i % 500 == 0:
            print(f"processing {i}/{len(target_vocab)} target tokens")
        target_token = target_inverted_vocab[i]
        target_to_source_token_ids = torch.LongTensor(source_tokenizer([target_token], add_special_tokens=False)["input_ids"][0])
            
        target_to_source_token_ids = target_to_source_token_ids.to(gpu_device)
        if i < len(source_vocab) + 100:
            print("target_token", target_token)
            print("sub_tokens", source_tokenizer.tokenize(target_token, add_special_tokens=False))
            print("target_to_source_token_ids", target_to_source_token_ids)
        
        target_to_source_input_embedding = (
            source_input_embeddings.weight[target_to_source_token_ids]
            .mean(dim=0)
            .unsqueeze(dim=0)
            .cpu()
            .detach()
            .numpy()
        )
        target_to_source_output_embedding = (
            source_output_embeddings.weight[target_to_source_token_ids]
            .mean(dim=0)
            .unsqueeze(dim=0)
            .cpu()
            .detach()
            .numpy()
        )
        
        input_embeddings = np.concatenate((input_embeddings, target_to_source_input_embedding), axis=0)
        output_embeddings = np.concatenate((output_embeddings, target_to_source_output_embedding), axis=0)
    
    source_model = source_model.to(cpu_device)
    assert isinstance(source_model, GemmaForCausalLM)
    
    # Resize the model embeddings
    source_model.resize_token_embeddings(new_num_tokens=len(target_vocab))
    source_model.model.embed_tokens.weight.data = torch.Tensor(input_embeddings)
    source_model.lm_head.weight.data = torch.Tensor(output_embeddings)
    
    source_model.half()
    source_model.save_pretrained(save_directory=new_tokenizer_dir)
    
    
def train_tokenizer(
    in_file: str = 'vi_clean_corpus.txt',
    sp_model_name: str = 'vi-tokenizer-10k',
    max_sentence_length: int = 100000,
    vocab_size: int = 10000,
    model_type: str = "BPE"):
    
    spm.SentencePieceTrainer.train(
        input=in_file,
        model_prefix=sp_model_name,
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
        max_sentence_length=max_sentence_length,
        model_type=model_type,
        vocab_size=vocab_size,
        split_digits=True,
        split_by_unicode_script=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
    )

    # Load the trained SentencePiece model
    sp = spm.SentencePieceProcessor()
    model_file = sp_model_name + '.model'
    sp.load(model_file)

    # Encode a sample text using the trained tokenizer
    

def download_wikidata(
    data_path: str = 'TranCongThien/viet_wiki_data',
    saved_path: str = 'vi_clean_corpus.txt'):
    
    import datasets
    from tqdm import tqdm
    
    dataset = datasets.load_dataset(data_path, split='train')
    with open(saved_path, 'w', encoding='utf-8') as f_write:
        for item in tqdm(dataset, desc=f"Write data to {saved_path}"):
            text = item['text']
            f_write.write(text + '\n')
    print("Successfully download the dataset")
    

def test_new_bpe_tokenizer(sp_model_name: str = 'vi-tokenizer-10k'):
    # Load the trained SentencePiece model
    text = "Ronaldo đảm nhận băng đội trưởng của đội tuyển quốc gia vào tháng 7 năm 2008."
    text += " Năm 2015, Ronaldo được Liên đoàn bóng đá Bồ Đào Nha bầu chọn là cầu thủ Bồ Đào Nha xuất sắc nhất mọi thời đại."
        
    sp = spm.SentencePieceProcessor()
    model_file = sp_model_name + '.model'
    sp.load(model_file)
    print(sp.encode_as_pieces(text))
        
def test_new_gemma_tokenizer(vi_gemma_tokenizer_path: str = "initial-vi-gemma-2b"):
    text = "Ronaldo đảm nhận băng đội trưởng của đội tuyển quốc gia vào tháng 7 năm 2008."
    text += " Năm 2015, Ronaldo được Liên đoàn bóng đá Bồ Đào Nha bầu chọn là cầu thủ Bồ Đào Nha xuất sắc nhất mọi thời đại."

    new_tokenizer = GemmaTokenizer.from_pretrained(vi_gemma_tokenizer_path)
    llama_tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b")
    
    print("Original LLama Tokenizer: ")
    print(llama_tokenizer.tokenize(text))
    
    print("Our new tokenizer: ")
    print(new_tokenizer.tokenize(text))
        
if __name__ == '__main__':
    fire.Fire()