import yaml
import json
import mlrun
from mlrun.execution import MLClientCtx
from os import path, makedirs, listdir
from utils8 import data_collator_tensordataset, load_data, CustomTrainer, get_metrics, upload_model
from transformers import AutoModelForSequenceClassification, TrainingArguments, EvalPrediction, AutoTokenizer, set_seed, Trainer


@mlrun.handler()
def start_train(context: MLClientCtx):
    """
    Launch the training of the models.
    """

    # Load the configurations for the model training phase
    model_metadata_path = "../../../metadata/model.yml"
    with open(model_metadata_path, 'r') as model_md_content:
        models_metadata = yaml.safe_load(model_md_content)
    model_metadata = models_metadata["models"][0]["training"]

    output_dir = model_metadata["output_dir"]
    bucket_name = model_metadata["bucket_name"]
    lang = model_metadata["parameters"]["language"]
    data_path = model_metadata["data"]["reference"]
    models_path = model_metadata["output_dir"]
    epochs = model_metadata["parameters"]["epochs"]
    batch_size = model_metadata["parameters"]["batch_size"]
    learning_rate = 3e-5 # float(model_metadata["parameters"]["learning_rate"])
    max_grad_norm = model_metadata["parameters"]["max_grad_norm"]
    threshold = float(model_metadata["parameters"]["threshold"])
    eval_metric = model_metadata["parameters"]["eval_metric"]
    full_metrics = model_metadata["parameters"]["full_metrics"]
    class_report_step = model_metadata["parameters"]["class_report_step"]
    save_class_report = model_metadata["parameters"]["save_class_report"]
    seeds = model_metadata["parameters"]["seeds"]
    device = model_metadata["parameters"]["device"]
    custom_loss = model_metadata["parameters"]["custom_loss"]
    weighted_loss = model_metadata["parameters"]["weighted_loss"]
    trust_remote = model_metadata["parameters"]["trust_remote"]
    fp16 = model_metadata["parameters"]["fp16"]
    report_to = model_metadata["parameters"]["report_to"]
    
    
    # Load the configuration for the models of all languages
    with open("config/models.yml", "r") as config_fp:
        config = yaml.safe_load(config_fp)

    # Load the seeds for the different splits
    if seeds != "all":
        seeds = seeds.split(",")
    else:
        seeds = [name.split("_")[1] for name in listdir(path.join(data_path, lang)) if "split" in name]

    print(f"Working on device: {device}")

    # Create the directory for the models
    if not path.exists(models_path):
        makedirs(models_path)

    global language
    language = lang

    print(f"\nTraining for language: '{lang}' using: '{config[lang]}'...")
    print(output_dir)
    print(seeds)

    # Train the models for all splits
    for seed in seeds:
        global current_split
        current_split = seed

        # Load the data
        train_set, dev_set, num_classes = load_data(data_path, lang, "train", seed)

        # Create the directory for the models of the current language
        makedirs(path.join(models_path, lang,
                    seed), exist_ok=True)

        # Create the directory for the classification report of the current language
        if save_class_report:
            makedirs(path.join(models_path, lang, str(
                seed), "train_reports"), exist_ok=True)

        set_seed(int(seed))

        tokenizer = AutoTokenizer.from_pretrained(config[lang])

        with open(path.join(data_path, language, f"split_{seed}", "train_labs_count.json"), "r") as weights_fp:
            data = json.load(weights_fp)
            labels = list(data["labels"].keys())

        model = AutoModelForSequenceClassification.from_pretrained(
            config[lang],
            problem_type="multi_label_classification",
            num_labels=num_classes,
            id2label={id_label:label for id_label, label in enumerate(labels)},
            label2id={label:id_label for id_label, label in enumerate(labels)},
            trust_remote_code=trust_remote,
        )

        # If the device specified via the arguments is "cpu", avoid using CUDA
        # even if it is available
        no_cuda = True if device == "cpu" else False

        # Create the training arguments.
        train_args = TrainingArguments(
            path.join(models_path, lang, seed),
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            num_train_epochs=epochs,
            lr_scheduler_type="linear",
            warmup_steps=len(train_set),
            logging_strategy="epoch",
            logging_dir=path.join(
                models_path, lang, seed, 'logs'),
            save_strategy="epoch",
            no_cuda=no_cuda,
            seed=int(seed),
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model=eval_metric,
            optim="adamw_torch",
            optim_args="correct_bias=True",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            report_to=report_to,
            fp16=fp16,
        )

        def compute_metrics(p: EvalPrediction):
            """
            Compute the metrics for the predictions during the training.

            :param p: EvalPrediction object.
            :return: Dictionary with the metrics.
            """
            preds = p.predictions[0] if isinstance(
                p.predictions, tuple) else p.predictions
            result = get_metrics(p.label_ids, preds, models_path, threshold, save_class_report, class_report_step, full_metrics, eval_metric)
            return result

        # Create the trainer. It uses a custom data collator to convert the
        # dataset to a compatible dataset.

        if custom_loss:
            trainer = CustomTrainer(
                model,
                train_args,
                train_dataset=train_set,
                eval_dataset=dev_set,
                tokenizer=tokenizer,
                data_collator=data_collator_tensordataset,
                compute_metrics=compute_metrics
            )

            trainer.prepare_labels(
                data_path, lang, seed, device)

            if weighted_loss:
                trainer.set_weighted_loss()
        else:
            trainer = Trainer(
                model,
                train_args,
                train_dataset=train_set,
                eval_dataset=dev_set,
                tokenizer=tokenizer,
                data_collator=data_collator_tensordataset,
                compute_metrics=compute_metrics
            )
        trainer.train()
        trainer.save_model(output_dir)
        upload_model(bucket_name, output_dir)

        print(f"Best checkpoint path: {trainer.state.best_model_checkpoint}")

