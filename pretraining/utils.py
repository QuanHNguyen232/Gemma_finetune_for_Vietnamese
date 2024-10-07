import random, os, json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()
    
def set_seed(seed):
    """Set the random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def setup_distributed_training():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")

def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()
    
def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)

def unwrap_model(model):
    """Recursively unwraps a model from potential containers (as used in distributed training).
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model
    
def save_model_checkpoint_and_loader(
    model,
    tokenizer,
    optimizer,
    lr_scheduler,
    train_dataloader,
    rank,
    step,
):
    """saving model via rank0 cpu streaming and full_state_dict"""
    fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    # Pull optimizer state to rank 0
    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()
        cpu_state = {key: value.cpu() for key, value in cpu_state.items()}

        print(f"saving process: rank {rank}  done w model state_dict\n")
   

    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        folder_name = 'checkpoints/step-{}'.format(step)
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("save_dir: ", save_dir)
        
        optimizer_name = "optimizer.pt"
        save_optimizer_path = str(save_dir) + "/" + optimizer_name
        torch.save(optim_state, save_optimizer_path)
        print(f"--> saved {save_optimizer_path} to {save_dir}")
        
        lr_scheduler_name = "lr_scheduler.pt"
        save_schedular_path = str(save_dir) + "/" + lr_scheduler_name
        torch.save(lr_scheduler.state_dict(), save_schedular_path)
        print(f"--> saved {save_schedular_path} to {save_dir}")

        # save model
        unwrap_model(model).save_pretrained(
            save_dir,
            state_dict=cpu_state,
            safe_serialization=True,
        )
        # save tokenizer
        tokenizer.save_pretrained(save_dir) # Lưu tokenizer cùng với mô hình
        
        state_dict_name = f"train_loader_state_dict.json"
        save_loader_path = str(save_dir) + "/" + state_dict_name
        with open(save_loader_path, 'w') as f:
            json.dump(train_dataloader.state_dict(), f, indent=4, ensure_ascii=False)
        
        print(f"data loader saved for step {step} at {save_loader_path}\n")