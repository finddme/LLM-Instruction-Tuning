import logging
import re, os, gc
from datasets import Dataset, load_dataset
import numpy as np
from transformers import Pipeline, PreTrainedTokenizer
from accelerate import infer_auto_device_map
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig, 
    DataCollatorForLanguageModeling, LlamaTokenizer, LlamaForCausalLM)
import os
from functools import partial
from typing import Any, Dict, List, Tuple, Union
from huggingface_hub import login

logger = logging.getLogger(__name__)

# login("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")

# output_dir="openlm_research"
# # model_n="EleutherAI/polyglot-ko-1.3b"
# model_n="openlm-research/open_llama_13b"
# # model_n="databricks/dolly-v2-12b"
# data_n="alpaca_and_dolly.jsonl"

# output_dir="KoAlpaca_Polyglot"
output_dir="Polyglot_12.8"
# model_n="EleutherAI/polyglot-ko-1.3b"
model_n="beomi/KoAlpaca-Polyglot-12.8B"
# model_n="databricks/dolly-v2-12b"
data_n="alpaca_and_dolly.jsonl"



INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}"
DEFAULT_SEED = 42

PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]
    
tokenizer = AutoTokenizer.from_pretrained(model_n)
# tokenizer = LlamaTokenizer.from_pretrained(model_n)
tokenizer.pad_token = tokenizer.eos_token
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42
tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})

model = AutoModelForCausalLM.from_pretrained(model_n, 
                                             device_map="auto", ########################################################################################################################
                                             torch_dtype=torch.bfloat16,
                                             use_cache=True)###################################################

# model = LlamaForCausalLM.from_pretrained(model_n, 
#                                              device_map="auto", ########################################################################################################################
#                                              load_in_8bit=True)
# model = AutoModelForCausalLM.from_pretrained(model_n, 
#                                              device_map="auto", ########################################################################################################################
#                                              torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))

from datasets import load_dataset
def load_training_dataset(path_or_dataset: str = data_n) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    #dataset = load_dataset("databricks/databricks-dolly-15k")["train"]
    dataset = load_dataset('json', data_files=f"./{path_or_dataset}")['train']
    logger.info("Found %d rows", dataset.num_rows)

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")

        # if not instruction:
        #     raise ValueError(f"Expected an instruction in: {rec}")

        # if not response:
        #     raise ValueError(f"Expected a response in: {rec}")

        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
        else:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
        return rec

    dataset = dataset.map(_add_text)

    return dataset


def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length=512) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset()

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    logger.info("Processed dataset has %d rows", dataset.num_rows)
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info("Processed dataset has %d rows after filtering for truncated records", dataset.num_rows)

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset

"""## Finetuning Dolly"""

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, GPTJForCausalLM, set_seed

from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

# Settings for A100 - For 3090 
MICRO_BATCH_SIZE = 4  # change to 4 for 3090
BATCH_SIZE = 32
# GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
world_size = int(os.environ.get("WORLD_SIZE", 1))
gradient_accumulation_steps = BATCH_SIZE // MICRO_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = gradient_accumulation_steps // world_size
EPOCHS = 999  # paper uses 3
LEARNING_RATE = 2e-5  
# LEARNING_RATE = 5e-6
CUTOFF_LEN = 256  
# LORA_R = 4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

model = prepare_model_for_int8_training(model, 
                                        use_gradient_checkpointing=True)
seed=DEFAULT_SEED
set_seed(seed)
processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=512, seed=seed)
split_dataset = processed_dataset.train_test_split(test_size=0.002, seed=seed)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

# data = load_dataset("json", data_files=f"/workspace/dolly/dolly/data/{data_n}")

'''
data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)
'''



# logger.info("Preprocessing dataset")
# _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer)
# data = data.map(
#     _preprocessing_function,
#     batched=True,
#     remove_columns=["instruction", "context", "response","text", "category"],
# )

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)
        #print("####################################LABLES?????????????????")
        labels = batch["labels"].clone()
        #print(labels)
        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch

data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

model.is_parallelizable = True
model.model_parallel = True
save_steps=100

trainer = transformers.Trainer(
    model=model,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],

    tokenizer=tokenizer,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        fp16=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=False,###################################################
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        output_dir=output_dir,
        report_to="wandb",
        evaluation_strategy="steps",###################################################
        eval_steps=50,######################################
        save_total_limit=8,
        save_strategy="steps",
        save_steps=save_steps,
    ),
    # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)#,return_tensors="pt", pad_to_multiple_of=8),
    data_collator=data_collator
)
model.config.use_cache = False
# trainer.train(resume_from_checkpoint=False)
trainer.train()
torch.cuda.empty_cache() 
gc.collect()
# trainer.train(resume_from_checkpoint=resume_from_checkpoint)
trainer.save_model(output_dir=output_dir)
model.save_pretrained(output_dir)
trainer.save_state()
# trainer.save_model(output_dir="lora-dolly")


print("DONE")


######################################################################################  adapters_weights -> LORA_Adapter_weights_save
# from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
# BASE_MODEL = model_n
# OUTPUT_DIR = f"{output_dir}/"
# STATE_DICT = f"{output_dir}/checkpoint-{save_steps}/pytorch_model.bin"

# model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# # This needs to match your training configuration _exactly_.
# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     r=64,
#     lora_alpha=32,
#     lora_dropout=0.05,
# )
# model = get_peft_model(model, peft_config)

# adapters_weights = torch.load(STATE_DICT, map_location="cpu")
# set_peft_model_state_dict(model, adapters_weights)

# model.save_pretrained(OUTPUT_DIR)
######################################################################################  adapters_weights (필요한 ck마다 만들어 사용)


