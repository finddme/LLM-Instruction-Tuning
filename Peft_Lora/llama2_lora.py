import torch
# torch.cuda.is_available()

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from typing import List

import fire, gc
import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import login
import os
login("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")
data_path="KIDA_LLAMA_traindata_total.jsonl"
training_data = load_dataset("json", data_files=data_path,
                             split="train"
                            )
print(f"trainset : {len(training_data)}")
torch.cuda.empty_cache() 
gc.collect()

llama_base_model_name = "./LDCC-Instruct-Llama-2-ko-13B-v1.4"
optimized_llama_model = "LLAMA2_13b_total2"

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(llama_base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
llama_base_model = AutoModelForCausalLM.from_pretrained(
    llama_base_model_name,
    quantization_config=quant_config,
    # torch_dtype=torch.float16,
    # device_map={"":1}
    device_map="auto"
    # device_map={'':torch.cuda.current_device()}
    )

print(f"1 {llama_base_model.device}")
torch.cuda.empty_cache() 
gc.collect()

llama_base_model.config.use_cache = False
llama_base_model.config.pretraining_tp = 1

# LoRA Config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    # r=8,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

print(f"2 {llama_base_model.device}")
torch.cuda.empty_cache() 
gc.collect()

# Training Params
training_params = TrainingArguments(
    output_dir="./output/"+optimized_llama_model,
    num_train_epochs=12,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    # optim="paged_adamw_32bit",
    optim="adamw_torch",
    save_steps=10,
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
#     evaluation_strategy="steps",
#     eval_steps=25,
    lr_scheduler_type="constant",
    report_to="wandb",
    save_total_limit=8
)
def formatting_func(example):
    text = [f"""
    ### instruction: {example['instruction']}\n 
    ### context: {example['context']}\n 
    ### response: {example['response']}\n 
    """]
    return text

print(f"3 {llama_base_model.device}")

torch.cuda.empty_cache() 
gc.collect()

# Trainer
llama_fine_tuning = SFTTrainer(
    model=llama_base_model,
    train_dataset=training_data,
    peft_config=peft_config,
    # dataset_text_field="text",
    formatting_func=formatting_func,
    tokenizer=llama_tokenizer,
    args=training_params,
    max_seq_length=512,
)

print(f"4 {llama_base_model.device}")

torch.cuda.empty_cache() 
gc.collect()

# Training
llama_fine_tuning.train()

# Save Model
llama_fine_tuning.model.save_pretrained(optimized_llama_model)