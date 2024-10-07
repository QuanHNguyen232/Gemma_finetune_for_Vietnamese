"""Datasets for converting to MDS Shards."""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
from argparse import ArgumentParser, Namespace
from enum import Enum
from glob import glob
from typing import Dict, Iterable, Optional, Union, Any

import numpy as np
import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import psutil, platform
from composer.utils import dist


def build_tokenizer(
        tokenizer_name: str,
        tokenizer_kwargs: Dict[str, Any]) -> PreTrainedTokenizerBase:
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    signal_file_path = f'.node_{dist.get_node_rank()}_local_rank0_completed_tokenizer_setup'

    if dist.is_available() and dist.is_initialized(
    ) and dist.get_world_size() > 1:
        # Make sure the tokenizer files are downloaded and cached first by local rank 0
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            pass

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                **tokenizer_kwargs)
    tokenizer.model_max_length = tokenizer_kwargs.get(
        'model_max_length',
        int(1e30),
    )

    if dist.is_available() and dist.is_initialized(
    ) and dist.get_world_size() > 1:
        if dist.get_local_rank() == 0:
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_tokenizer_setup')

        dist.barrier()

        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)

    return tokenizer


def build_dataloader(dataset: Dataset, batch_size: int,
                     num_workers: Optional[int]) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if 'linux' or 'macos' in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    
def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


class ConcatTokensDataset(IterableDataset):
    """IterableDataset that yields tokenized text samples."""
    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        eos_text: str,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_text = eos_text
        self.should_wrap = True

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        assert len(self.eos_tokens) == 1, 'eos_text must be a single token'
        eos_text_provided = self.eos_text != ''

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        for sample in self.hf_dataset:
            text = sample['text']
            encoded = self.tokenizer(text,
                                     truncation=False,
                                     padding=False,
                                     add_special_tokens=False)
            iids = encoded['input_ids']
            buffer = buffer + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(concat_sample).tobytes()
                }

def build_hf_dataset(
    path: str,
    max_length: Optional[int] = None,
    eos_text: str = '',
    tokenizer: PreTrainedTokenizerBase = None,
) -> IterableDataset:
    
    hf_dataset = hf_datasets.load_dataset(path, split='train', streaming=True)
    
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(
            f'{tokenizer=} must be of type PreTrainedTokenizerBase')
    if max_length is None:
        raise ValueError(f'max_length must be set.')
    if eos_text == '':
        raise ValueError(f'eos_text must be set.')
    dataset = ConcatTokensDataset(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        eos_text=eos_text)
    
    return dataset

            

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default='zstd')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--concat_tokens', type=int, default=4096)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--eos_text', type=str, required=False, default="</s>")

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.split))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed
    

def main(args: Namespace) -> None:
    tokenizer = build_tokenizer(args.tokenizer, {})
    columns = {'tokens': 'bytes'}
    
    dataset = build_hf_dataset(path=args.path,
                               max_length=args.concat_tokens,
                               eos_text=args.eos_text,
                               tokenizer=tokenizer)
    
    loader = build_dataloader(
        dataset=dataset,
        batch_size=512,
        num_workers=16,
    )
     
    samples = generate_samples(
        loader,
        truncate_num_samples=None
    )
    # Write samples
    print("Building wikidataset...")
    print(f'Converting to MDS format...')
    with MDSWriter(columns=columns,
                   out=os.path.join(args.out_root),
                   compression=args.compression) as out:
        for sample in tqdm(samples, desc=args.out_root):
            out.write(sample)

if __name__ == '__main__':
    args = parse_args()
    main(args)