import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = 'gemma-2b-lora-chat'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map="cuda",
    use_cache=True,
)

eos_token_id = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<eos>"])
print("eos_token_id: ", eos_token_id)

conversation = []
print("Enter 'reset' to clear the chat history.")
while True:
    human = input("Human:")
    if human.lower() == "reset":
        conversation = []
        print("The chat history has been cleared!")
        continue

    conversation.append({"role": "user", "content": human })
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True).to(model.device)
    
    out_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        top_k=40,
        temperature=0.1,
        repetition_penalty=1.05,
        eos_token_id=eos_token_id,
    )
    assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
    print("Assistant: ", assistant) 
    conversation.append({"role": "assistant", "content": assistant })