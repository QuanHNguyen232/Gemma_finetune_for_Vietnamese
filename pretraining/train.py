import os
import fire
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.optim as optim 

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from dataset import get_streaming_data
from fsdp import fsdp_auto_wrap_policy, apply_fsdp_checkpointing, get_policies
from utils import set_seed, clear_gpu_cache, setup_distributed_training, cleanup, setup_environ_flags, save_model_checkpoint_and_loader

def load_model(model_path, offload_params: bool = False, resume_from_checkpoint: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if (not resume_from_checkpoint):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=None,
            use_cache=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resume_from_checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=None,
            use_cache=False)
            
    model = FSDP(
        model,
        auto_wrap_policy=get_policies(),
        cpu_offload=CPUOffload(offload_params=offload_params),
        mixed_precision=None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
    )
    
    apply_fsdp_checkpointing(model)
    
    return model, tokenizer

def load_optimizer(model, num_steps):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,
        betas=(0.9, 0.95),
        eps=1e-05,
        weight_decay=0.05,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)
    return optimizer, scheduler

def resume_from_checkpoint(model, optimizer, scheduler, resume_from_checkpoint: str):
    optimizer_checkpoint_path = resume_from_checkpoint + "/" + "optimizer.pt"
    full_osd = None
    
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)
    
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
    optimizer.load_state_dict(sharded_osd)
    
    lr_scheduler_path = resume_from_checkpoint + "/" + "lr_scheduler.pt"
    scheduler_state_dict = torch.load(lr_scheduler_path)
    scheduler.load_state_dict(scheduler_state_dict)
    
    return optimizer, scheduler
    
def train(
    model,
    tokenizer,
    train_dataloader,
    optimizer,
    lr_scheduler,
    batch_size,
    gradient_accumulation_steps,
    save_step,
    local_rank
    ):
    
    # Set up progress bar
    total_length = len(train_dataloader)//gradient_accumulation_steps
    pbar = tqdm(colour="blue", total=total_length, dynamic_ncols=True)
    next_step = int(train_dataloader.state_dict().get('sample_in_epoch', 0))//(int(os.environ["WORLD_SIZE"]) * batch_size)
    pbar.update(next_step)
    
    # Training
    total_loss = 0.0
    for step, batch in enumerate(train_dataloader, start=next_step):
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        loss = model(**batch).loss
        loss = loss / gradient_accumulation_steps
        total_loss += loss.detach().float()
        
        loss.backward()
        model.clip_grad_norm_(1.0)
        
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)

        pbar.set_description(f"Lr: {lr_scheduler.get_lr()} | Step {step}/{len(train_dataloader)} Completed (loss: {loss.detach().float()})")
        
        if (step % save_step == 0 and step != 0) or step + 1 == len(train_dataloader):
            save_model_checkpoint_and_loader(model,tokenizer, optimizer, lr_scheduler, train_dataloader, local_rank, step)

def main(
    model_path: str,
    local_data_path: str,
    offload_params: bool = False,
    resume_from_checkpoint: str = None,
    batch_size: int = 2,
    num_workers: int = 4,
    context_length: int = 4096,
    gradient_accumulation_steps: int = 1):
    
    set_seed(42)
    setup_distributed_training()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    
    model, tokenizer = load_model(model_path, offload_params=offload_params, resume_from_checkpoint=resume_from_checkpoint)
    train_dataloader = get_streaming_data(local_path=local_data_path, batch_size=batch_size, num_workers=num_workers, context_length=context_length)
    optimizer, lr_scheduler = load_optimizer(model, len(train_dataloader))
    
    if resume_from_checkpoint:
        optimizer, scheduler = resume_from_checkpoint(model, optimizer, scheduler, resume_from_checkpoint)
    
    train(
        model,
        tokenizer,
        train_dataloader,
        optimizer,
        lr_scheduler,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_step=30000,
        local_rank=local_rank,
    )

if __name__ == "__main__":
    fire.Fire(main)
    
    
        
    
    
    
    