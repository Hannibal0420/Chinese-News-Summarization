import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    Adafactor
)
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

parser = argparse.ArgumentParser(description='Run validation.')

# Add arguments
parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
parser.add_argument('--validation_dataset', type=str, required=True, help='Path to the validation dataset')
parser.add_argument('--output_file_name', type=str, required=True, help='Path to the output file')

# Execute the parse_args() method
args = parser.parse_args()

accelerator = Accelerator()
#accelerator.wait_for_everyone()

data_files = {}
data_files["validation"] = args.validation_dataset
raw_datasets = load_dataset('json', data_files=data_files)
model_path = args.model_path
config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=not False, trust_remote_code=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, from_tf=bool(".ckpt" in model_path), config=config, trust_remote_code=False)

# We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
# on a small vocab and want a smaller embedding size, remove this test.
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))
if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

prefix = "summarize: "
column_names = raw_datasets["validation"].column_names
text_column = 'maintext'
summary_column = 'title'
id_column = 'id'
val_max_target_length = 128
padding = False

def preprocess_function(examples):
    inputs = examples[text_column]
    ids = examples[id_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)
    int_ids = [int(id_str) for id_str in ids]
    model_inputs["id"] = int_ids
    return model_inputs

with accelerator.main_process_first():
    # Temporarily set max_target_length for validation.
    max_target_length = val_max_target_length
    eval_dataset = raw_datasets["validation"].map(
        preprocess_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=not False,
        desc="Running tokenizer on dataset",
    )

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if accelerator.use_fp16 else None,
)

def postprocess_text(preds):
    preds = [pred.strip() for pred in preds]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

    return preds

eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=8)

# Prepare everything with our `accelerator`.
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)


model.eval()

gen_kwargs = {
    "max_length": val_max_target_length,
    "num_beams": 3,
}

decoded_preds_list = []
preds_ids_list = []

for step, batch in enumerate(eval_dataloader):
    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        generated_tokens = accelerator.gather_for_metrics((generated_tokens))
        generated_tokens = generated_tokens.cpu().numpy()

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_preds = postprocess_text(decoded_preds)
        
        decoded_preds_list.extend(decoded_preds)
        preds_ids_list.extend(batch["id"].tolist())


with open(args.output_file_name, 'w', encoding='utf-8') as file:
    # Iterate over both lists simultaneously
    for title, id in zip(decoded_preds_list, preds_ids_list):
        # Create a dictionary for the current title and id
        data = {"title": title, "id": id}
        # Write the dictionary to the file as a JSON-formatted string
        file.write(json.dumps(data, ensure_ascii=False) + '\n')


accelerator.end_training()
