import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import bitsandbytes as bnb
import wandb
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

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
print("accelerator")
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

data_path="KIDA_Mistral_traindata_800.jsonl"
login("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")
# Defined in the secrets tab in Google Colab
training_data = load_dataset("json", data_files=data_path,
                             split="train"
                            )

print("training_data:",len(training_data))
training_data=training_data.shuffle(seed=42)


model_name = "./OpenHermes-2.5-Mistral-7B"
optimized_model = "mistral"


def chatml_format(example):
    # Format system
    if len(example['context']) > 0:
        message = {"role": "system", "content": "Below is an instruction(question) that describes a task. Write a response that appropriately completes the request, referring to given Context."+ example['context']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": example['instruction']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = example['response'] + "<|im_end|>\n"

    # Format rejected answer
    rejected = example['rejected'] + "<|im_end|>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Load dataset
training_data = load_dataset("json", data_files=data_path,
                             split="train"
                            )

# Save columns
original_columns = training_data.column_names

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Format dataset
training_data = training_data.map(
    chatml_format,
    remove_columns=original_columns
)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"": Accelerator().local_process_index},
#     device_map=torch.device("cuda:1"),
#     max_memory={0: "80GiB", 1: "80GiB", "cpu": "30GiB"}
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model.config.use_cache = False
model = accelerator.prepare_model(model)
# Reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": Accelerator().local_process_index},
#     device_map=torch.device("cuda:1"),
    torch_dtype=torch.float16,
    load_in_4bit=True
)
ref_model = accelerator.prepare_model(ref_model)
# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
#     num_train_epochs=3,
    # save_strategy="no",
    save_steps=10,
    logging_steps=1,
    output_dir="./output/"+optimized_model,
#     optim="paged_adamw_32bit",
    optim="adamw_torch",
    warmup_steps=100,
#     bf16=True,
    bf16=False,
    report_to="wandb",
    save_total_limit=8
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model,
    ref_model,
    args=training_args,
    train_dataset=training_data,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)

# Fine-tune model with DPO
dpo_trainer.train()

dpo_trainer.model.save_pretrained(optimized_llama_model)