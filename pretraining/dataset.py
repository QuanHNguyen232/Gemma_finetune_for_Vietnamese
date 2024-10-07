from typing import Dict, Any, List
import fire
import torch
import numpy as np
from streaming import StreamingDataset, StreamingDataLoader

def _read_binary_tokenized_sample(sample: Dict[str, Any], max_seq_len: int = 4096) -> torch.Tensor:
    return torch.from_numpy(
        np.frombuffer(sample['tokens'], dtype=np.int64)[:max_seq_len].copy())

def intermediate_collate_fn(batch: List[Dict[str, Any]], max_seq_len: int) -> Dict[str, Any]:
    return {'input_ids': torch.stack([_read_binary_tokenized_sample(sample, max_seq_len) for sample in batch])}

def combined_collate_fn(batch: List[Dict[str, Any]], max_seq_len: int = 4096) -> Dict[str, Any]:
    intermediate_result = intermediate_collate_fn(batch, max_seq_len)

    attention_mask = (intermediate_result['input_ids'] != 0).long()

    labels = intermediate_result['input_ids'].clone()

    result = {
        'input_ids': intermediate_result['input_ids'],
        'attention_mask': attention_mask,
        'labels': labels
    }
    return result

def get_streaming_data(local_path, batch_size: int = 4, num_workers: int = 4, context_length: int = 4096):
    train_dataset = StreamingDataset(
        local=local_path,
        batch_size=batch_size,
        shuffle=True,
        shuffle_seed=42
    )
    
    train_dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: combined_collate_fn(b, max_seq_len=context_length),
    )
    
    return train_dataloader

def test_streaming_data(
    index: int,
    local_data_path: str = 'vi-wiki',
    tokenizer_path: str = 'initial-vi-gemma-2b'
):
    from transformers import AutoTokenizer
    dataset = StreamingDataset(
        local=local_data_path,
        shuffle=True,
        shuffle_seed=42
    )
    input_ids = _read_binary_tokenized_sample(dataset[index])
    print(input_ids)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print([tokenizer.decode(input_ids)])
    

if __name__ == "__main__":
    fire.Fire()