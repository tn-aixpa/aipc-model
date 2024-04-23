import os
import torch
import json
from math import ceil
from pathlib import Path
from trl import SFTTrainer
from peft import LoraConfig
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

########################################################## Preprocessing data

def load_jsonl(data_dir):
    data_path = Path(data_dir).as_posix()
    data = load_dataset("json", data_files=data_path)
    return data


def save_jsonl(data_dicts, out_path):
    with open(out_path, "w") as fp:
        for data_dict in data_dicts:
            fp.write(json.dumps(data_dict) + "\n")


def load_data_sql(data_dir: str = "data_sql"):
    dataset = load_dataset("b-mc2/sql-create-context")

    dataset_splits = {"train": dataset["train"]}
    out_path = Path(data_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for key, ds in dataset_splits.items():
        with open(out_path, "w") as f:
            for item in ds:
                newitem = {
                    "input": item["question"],
                    "context": item["context"],
                    "output": item["answer"],
                }
                f.write(json.dumps(newitem) + "\n")
                

def get_train_val_splits(
    data_dir: str = "data_sql",
    val_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
):
    data = load_jsonl(data_dir)
    num_samples = len(data["train"])
    val_set_size = ceil(val_ratio * num_samples)

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=shuffle, seed=seed
    )
    return train_val["train"].shuffle(), train_val["test"].shuffle()


###########################################################################3


def formatting_func(example):
    text = f"You are a powerful text-to-SQL model. \
    Your job is to answer questions about a database. \
    You are given a question and context regarding one or more tables. \
    \n\nYou must output the SQL query that answers the question. \
    \n\n### Input:\n{example['input']} \
    \n\n### Context:\n{example['context']} \
    \n\n### Response:\n{example['output']}"
    return text


def training(context):
    login('')

    # setting quantization params
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model_id = "meta-llama/Llama-2-7b-hf"
    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Set it to a new token to correctly attend to EOS tokens.
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    # setting LoRA params
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.add_adapter(lora_config)
    # dump data to data_sql
    load_data_sql(data_dir="data_sql")
    raw_train_data, raw_val_data = get_train_val_splits(data_dir="data_sql")
    save_jsonl(raw_train_data, "train_data_raw.jsonl")
    save_jsonl(raw_val_data, "val_data_raw.jsonl")

    output_dir = f"./results"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 400
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 1000
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
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=raw_train_data,
        packing=True,
        dataset_text_field="id",
        tokenizer=tokenizer,
        max_seq_length=1024,
        formatting_func=formatting_func,
    )
    os.environ["WANDB_ENTITY"] = "llm"
    os.environ["WANDB_PROJECT"] = "llama2-7b-finetuned"
    os.environ["WANDB_API_KEY"] = ""

    trainer.train()
    