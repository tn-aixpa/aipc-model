import os
import torch
import json
import glob
from math import ceil
from pathlib import Path
from trl import SFTTrainer
from zipfile import ZipFile
from peft import LoraConfig, PeftModel
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

def save_full_model(model_name, output_dir, adapters_path):
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    # Initialize PeftModel with base model and adapters path
    model = PeftModel.from_pretrained(base_model, adapters_path)
    # Merge the trainer adapter with the base model and unload the adapter
    model = model.merge_and_unload()
    full_model_path = output_dir + "/full"
    os.makedirs(full_model_path, exist_ok=True)
    model.save_pretrained(full_model_path, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(full_model_path)
    with ZipFile("text2seq_full_model.zip", "w") as zip_file:
        for file in glob.glob(f"{full_model_path}/*"):
            zip_file.write(file)
    return "text2seq_full_model.zip"

def training(context):
    login(context.get_secret('HF_TOKEN'))

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
    os.environ["WANDB_ENTITY"] = context.get_secret('WANDB_ENTITY')
    os.environ["WANDB_PROJECT"] = context.get_secret('WANDB_PROJECT')
    os.environ["WANDB_API_KEY"] = context.get_secret('WANDB_API_KEY')

    trainer.train()
    adapters_path = [el for el in os.listdir(output_dir) if el.startswith('checkpoint')][0]
    adapters_path = os.path.join(output_dir, adapters_path)
    full_model_file = save_full_model(model_id, output_dir, adapters_path)
    # log model to MLRun
    context.log_model(
        "llama2_text2seq_model",
        parameters={
            "max_steps": 1000
        },
        metrics = {}, # TODO
        model_file=full_model_file,
        labels={"class": "AutoModelForCausalLM"},
        algorithm="AutoModelForCausalLM",
        framework="transformers"
    )
    
    