#!/bin/bash

python train_sweep.py \
  --model_name_or_path google/mt5-small \
  --train_file ../data/train_trim.jsonl \
  --validation_file ../data/public_trim.jsonl \
  --text_column maintext \
  --summary_column title \
  --num_beams 3 \
  --per_device_train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=1 \
  --source_prefix "summarize: " \
  --with_tracking \
  --report_to "wandb"
