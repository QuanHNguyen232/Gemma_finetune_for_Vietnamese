import argparse
import os
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import AutoTokenizer, LlamaTokenizer, MistralForCausalLM
import torch
import sentencepiece as spm
from datasets import load_dataset

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import json

import re

def has_non_alphabetic_chars(token):
    return any(not char.isalpha() for char in token)

def merge_tokenizer(
    source_tokenizer_dir,
    new_tokenizer_model,
    new_tokenizer_dir):
    
    source_tokenizer = LlamaTokenizer.from_pretrained(source_tokenizer_dir)
    source_sp_processor = source_tokenizer.sp_model
    source_spm = sp_pb2_model.ModelProto()
    source_spm.ParseFromString(source_sp_processor.serialized_model_proto())
    
    source_spm_tokens = set([p.piece for p in source_spm.pieces])
    
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(new_tokenizer_model)
    
    sp_tgt_pb2 = sp_pb2_model.ModelProto()
    sp_tgt_pb2.ParseFromString(sp_tgt.serialized_model_proto())
    new_tgt_tokens = list(set([p.piece for p in sp_tgt_pb2.pieces]))
    print("The number of original tokens:", len(source_spm_tokens))
    print("The number of new tokens:", len(new_tgt_tokens))
    
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
    
    os.makedirs(new_tokenizer_dir)
    target_tokenizer_model_path = os.path.join(new_tokenizer_dir, "tokenizer.model")
    with open(file=target_tokenizer_model_path, mode="wb") as fp:
        fp.write(source_spm.SerializeToString())
    
    target_tokenizer = LlamaTokenizer(vocab_file=target_tokenizer_model_path)
    target_tokenizer.save_pretrained(save_directory=new_tokenizer_dir)
    print("Vocab size", len(target_tokenizer))
    target_tokenizer.push_to_hub(repo_id='chiennv/'+ new_tokenizer_dir,
                                 token='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',
                                 private=True)
    print(f"Successfully save expand tokenizer to {new_tokenizer_dir}")
    
def reinit_model(model_name, new_tokenizer_dir):
    source_tokenizer = LlamaTokenizer.from_pretrained(model_name)
    source_tokenizer.add_bos_token = False
    source_tokenizer.add_eos_token = False
    if source_tokenizer.pad_token is None:
        source_tokenizer.pad_token = source_tokenizer.unk_token
    source_vocab = source_tokenizer.get_vocab()
    
    target_tokenizer = LlamaTokenizer.from_pretrained(new_tokenizer_dir)
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
    
    source_model = MistralForCausalLM.from_pretrained(model_name)
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
    
    for i in range(len(source_vocab), len(target_vocab)):
        if i % 500 == 0:
            print(f"processing {i}/{len(target_vocab)} target tokens")
        target_token = target_inverted_vocab[i]
        target_to_source_token_ids = torch.LongTensor(source_tokenizer([target_token], add_special_tokens=False)["input_ids"][0])
        
        if target_to_source_token_ids[0].item() == 28705: # ignore the prefix space
            target_to_source_token_ids = target_to_source_token_ids[1:]
            
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
    assert isinstance(source_model, MistralForCausalLM)
    
    # expand
    source_model.resize_token_embeddings(new_num_tokens=len(target_vocab))
    source_model.model.embed_tokens.weight.data = torch.Tensor(input_embeddings)
    source_model.lm_head.weight.data = torch.Tensor(output_embeddings)
    
    source_model.half()
    source_model.save_pretrained(save_directory=new_tokenizer_dir)
    source_model.push_to_hub(repo_id='chiennv/Vistral-base-init',
                             private=True,
                             token='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')
    target_tokenizer.push_to_hub(repo_id='chiennv/Vistral-base-init')
    
    
def train_tokenizer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', default='vi_clean_corpus.txt', type=str)
    parser.add_argument('--sp_model_name', default='vi_clean_10k', type=str)
    parser.add_argument('--max_sentence_length', default=100000, type=int)
    parser.add_argument('--vocab_size', default=10000, type=int)
    parser.add_argument('--model_type', default="BPE", type=str)

    args = parser.parse_args()
    print(args)

    spm.SentencePieceTrainer.train(
        input=args.in_file,
        model_prefix=args.sp_model_name,
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
        max_sentence_length=args.max_sentence_length,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        split_digits=True,
        split_by_unicode_script=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
    )

    # makes segmenter instance and loads the model file (m.model)
    sp = spm.SentencePieceProcessor()
    model_file = args.sp_model_name + '.model'
    sp.load(model_file)

    # encode: text => id
    print(sp.encode_as_pieces('Lãnh tụ Hồ Chí Minh!'))

if __name__ == '__main__':
    train_tokenizer()
    
    # merge_tokenizer(
    #     source_tokenizer_dir='viet-mistral-v0-original-tokenizer',
    #     new_tokenizer_model='vi_clean_10k.model',
    #     new_tokenizer_dir='f-tokenizer-mistral-vi-10k'
    # )
    
    # reinit_model(
    #     model_name='mistralai/Mistral-7B-v0.1',
    #     new_tokenizer_dir='f-tokenizer-mistral-vi-10k',
    # )