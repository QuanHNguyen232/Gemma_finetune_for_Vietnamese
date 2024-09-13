import fire, os
import torch
import datasets
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def load_model(model_name_or_path, max_seq_length):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        padding_side='right',
    )
    
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    local_rank = os.environ["LOCAL_RANK"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{local_rank}",
    )
    model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=16, 
        lora_alpha=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=0.0, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    return tokenizer, model

def load_dataset(tokenizer: AutoTokenizer, dataset_name_or_path: str, max_seq_length: int):
    
    def convert_conversation_to_input(examples):
        input_ids = [tokenizer.apply_chat_template(conversation, tokenize=True)[: max_seq_length] for conversation in examples['conversations']]
        return {'input_ids': input_ids}
        
    dataset = datasets.load_dataset(dataset_name_or_path, split='train[:]')
    dataset = dataset.map(convert_conversation_to_input, remove_columns=list(dataset.features), batched=True)
    
    dataset = dataset.train_test_split(test_size=500, shuffle=True)
    return dataset['train'], dataset['test']

def train(
    model_name_or_path = 'gemma-2b',
    train_batch_size: int = 4,
    max_seq_length: int = 2048,
):
    dataset_name = 'chiennv/mini-ultrachat'
    output_dir = "nor-checkpoints"
    
    # Load model and tokenizer
    tokenizer, model = load_model(model_name_or_path, max_seq_length)
    
    # Load dataset
    train_dataset, eval_dataset = load_dataset(tokenizer=tokenizer, dataset_name_or_path=dataset_name, max_seq_length=max_seq_length)
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user\n",
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_gpu_eval_batch_size=1,
        bf16=True,
        learning_rate=2e-4,
        optim="adamw_8bit",
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=2,
        ddp_find_unused_parameters=False,
        group_by_length=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_dir)

if __name__ == "__main__":
    # fire.Fire(train)
    fire.Fire(train)