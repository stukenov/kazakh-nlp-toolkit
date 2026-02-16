# -*- coding: utf-8 -*-

from datasets import load_dataset

dataset_name = "kz-transformers/multidomain-kazakh-dataset"
dataset = load_dataset(dataset_name, split="train")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ybelkada/falcon-7b-sharded-bf16"

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

from transformers import TrainingArguments

output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "adamw"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
)

from transformers import Trainer

max_seq_length = 512

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_arguments,
    tokenizer=tokenizer,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'labels': torch.stack([f['input_ids'] for f in data])}
)

trainer.train()

