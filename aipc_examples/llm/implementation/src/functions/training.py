import os
import torch
import transformers
from functools import reduce
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)

def preprocess(sample, tokenizer, max_tokens=512, add_eos_token=True):
    """
    Tokenize input samples
    """
    result = tokenizer(
        sample["text"],
        truncation=True,
        max_length=max_tokens,
        padding=False,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result

def get_model_parameter(param_name):
    # Load model metadata
    model_metadata_path = "../../../metadata/model.yml"
    with open(model_metadata_path, 'r') as model_md_content:
        models_metadata = yaml.safe_load(model_md_content)        
    model_metadata = models_metadata["models"][0]["training"]
    list_parameters = list(filter(lambda element: element['name'] == param_name, model_metadata["parameters"]))
    return list_parameters[0]["value"] if len(list_parameters) > 0 else None


def get_optimization_parameter(param_name, type="peft"):
    """
    Load optimization metadata
    """
    optimization_metadata_path = "../../../metadata/optimization.yml"
    with open(optimization_metadata_path, 'r') as optimization_md_content:
        optimization_metadata = yaml.safe_load(optimization_md_content)        
    optimization_metadata_quantization = optimization_metadata[0]["quantization"]
    optimization_metadata_peft = optimization_metadata[0]["peft"]
    if type=='peft':
        list_parameters = list(filter(lambda element: element['name'] == param_name, optimization_metadata_peft["parameters"]))
    elif type=='quantization':
        list_parameters = list(filter(lambda element: element['name'] == param_name, optimization_metadata_quantization["parameters"]))
    else:
        raise Exception("optimization type should be 'peft' or 'quantization'")  
    return list_parameters[0]["value"] if len(list_parameters) > 0 else None

def train():
    """
    Training function
    """
    # Load model metadata
    model_metadata_path = "../../../metadata/model.yml"
    with open(model_metadata_path, 'r') as model_md_content:
        models_metadata = yaml.safe_load(model_md_content)        
    model_metadata = models_metadata["models"][0]["training"] 
    save_model = model_metadata["output_dir"]
    file_train = model_metadata["data"]["train"]
    file_valid = model_metadata["data"]["valid"]

    seed = get_model_parameter("seed")
    base_model = get_model_parameter("base_model")
    log_steps = get_model_parameter("log_steps")
    eval_steps = get_model_parameter("eval_steps")
    save_steps = get_model_parameter("save_steps")
    warmup_steps = get_model_parameter("warmup_steps")
    per_device_batch_size = get_model_parameter("per_device_batch_size")
    gradient_accumulation_steps = get_model_parameter("gradient_accumulation_steps")
    max_epochs = get_model_parameter("max_epochs")
    learning_rate = get_model_parameter("learning_rate")
    max_tokens = get_model_parameter("max_tokens")
    gradient_checkpointing = get_model_parameter("gradient_checkpointing")
    group_by_length = get_model_parameter("group_by_length")
    resume_checkpoint = get_model_parameter("resume_checkpoint")
    wandb_entity = get_model_parameter("wandb_entity")
    wandb_project = get_model_parameter("wandb_project")

    os.environ["WANDB_ENTITY"] = wandb_entity
    os.environ["WANDB_PROJECT"] = wandb_project

    transformers.set_seed(seed)

    # Activate 4-bit precision base model loading
    use_4bit = get_optimization_parameter("load_in_4bit", type="quantization")
    bnb_4bit_compute_dtype = get_optimization_parameter("bnb_4bit_compute_dtype", type="quantization")
    bnb_4bit_quant_type = get_optimization_parameter("bnb_4bit_quant_type", type="quantization")
    use_nested_quant = get_optimization_parameter("bnb_4bit_use_double_quant", type="quantization")

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )
    model.config.pretraining_tp = 1
    SPECIAL_TOKENS_DICT = {"additional_special_tokens": ["<|endofturn|>"]}
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model.resize_token_embeddings(len(tokenizer))

    train_data = load_dataset("text", data_files={"train": file_train})
    valid_data = load_dataset("text", data_files={"train": file_valid})
    train_data = train_data["train"].map(
        lambda example: preprocess(example, tokenizer, max_tokens)
    )
    valid_data = valid_data["train"].map(
        lambda example: preprocess(example, tokenizer, max_tokens)
    )

    print("[Info] {} train samples".format(len(train_data)))
    print("[Info] {} valid samples".format(len(valid_data)))

    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

    # Setting up LoRA configuration
    lora_r = get_optimization_parameter("rank", type="peft")
    # Alpha parameter for LoRA scaling
    lora_alpha = get_optimization_parameter("lora_alpha", type="peft")
    # Dropout probability for LoRA layers
    lora_dropout = get_optimization_parameter("lora_dropout", type="peft")
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            disable_tqdm=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=max_epochs,
            learning_rate=learning_rate,
            bf16=True,
            tf32=True,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            logging_steps=log_steps,
            output_dir=save_model,
            group_by_length=group_by_length,
            gradient_checkpointing=gradient_checkpointing,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
        callbacks=callbacks,
        peft_config=peft_config,
        dataset_text_field="text",
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    # trainer.model.save_model(save_model)
    base_model_pretrained = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # get best checkpoint's path to get adapter_config.json
    checkpoint_name = [
        el for el in os.listdir(save_model) if el.startswith("checkpoint")
    ][0]

    model = PeftModel.from_pretrained(
        base_model_pretrained, save_model + "/" + checkpoint_name
    )
    model = model.merge_and_unload()

    save_dir = os.path.join(save_model, "merged_model")
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(save_dir)
