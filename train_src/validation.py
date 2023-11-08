import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
from tw_rouge import get_rouge
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
val_max_target_length = 128
padding = False

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[summary_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)

    labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
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

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=8)

# Prepare everything with our `accelerator`.
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)


model.eval()

gen_kwargs = {
    "max_length": val_max_target_length,
    "num_beams": 3,
}

decoded_preds_list = []
decoded_labels_list = []

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
        labels = batch["labels"]
        labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
        generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
        generated_tokens = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        decoded_preds_list.extend(decoded_preds)
        decoded_labels_list.extend(decoded_labels)
result = get_rouge(decoded_preds_list, decoded_labels_list)
result = {k: round(v['f'] * 100, 4) for k, v in result.items()}
print(result)

accelerator.end_training()
