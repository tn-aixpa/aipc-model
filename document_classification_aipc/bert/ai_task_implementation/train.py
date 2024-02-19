import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, EvalPrediction, AutoTokenizer, set_seed, Trainer
import yaml
from os import path, makedirs, listdir
from utils8 import sklearn_metrics_single, sklearn_metrics_full, data_collator_tensordataset, load_data, CustomTrainer
import json
from mlrun.execution import MLClientCtx

language = ""
current_epoch = 0
current_split = 0


def get_metrics(y_true, predictions, models_path, threshold=0.5, save_class_report=False, class_report_step=1, full_metrics=False, eval_metric="f1_micro"):
    """
    Return the metrics for the predictions.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions. Default: 0.5.
    :return: Dictionary with the metrics.
    """
    global current_epoch
    global language
    global current_split

    metrics, class_report, _ = sklearn_metrics_full(
        y_true,
        predictions,
        "",
        threshold,
        False,
        save_class_report,
    ) if full_metrics else sklearn_metrics_single(
        y_true,
        predictions,
        "",
        threshold,
        False,
        save_class_report,
        eval_metric=eval_metric,
    )

    if save_class_report:
        if current_epoch % class_report_step == 0:
            with open(path.join(
                models_path,
                language,
                str(current_split),
                "train_reports",
                f"class_report_{current_epoch}.json",
            ), "w") as class_report_fp:
                class_report.update(metrics)
                json.dump(class_report, class_report_fp, indent=2)

    current_epoch += 1

    return metrics


def compute_metrics(p: EvalPrediction, threshold, models_path):
    """
    Compute the metrics for the predictions during the training.

    :param p: EvalPrediction object.
    :return: Dictionary with the metrics.
    """
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
    result = get_metrics(p.label_ids, preds, models_path, threshold)
    return result


def start_train(
        context: MLClientCtx, 
        lang: str = "it",
        data_path: str ="data/",
        models_path: str = "models/",
        seeds: str = "all",
        device: str = "cpu",
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        max_grad_norm: int = 5,
        threshold: float = 0.5,
        custom_loss: bool = False,
        weighted_loss: bool = False,
        fp16: bool = False,
        eval_metric: str = "f1_micro",
        save_class_report: bool = False,
        class_report_step: int = 1,
        full_metrics: bool = False,
        trust_remote: bool = False
        ):
    """
    Launch the training of the models.
    :param lang: Language to train the model on.
    :param data_path:Path to the EuroVoc data.
    :param models_path: Save path of the models.
    :param seeds: Seeds to be used to load the data splits, separated by a comma (e.g. 110,221). Use 'all' to use all the data splits.
    :param device: Device to train on.
            choices=["cpu", "cuda"]
    :param epochs: Number of epochs to train the model.
    :param batch_size: Batch size of the dataset.
    :param learning_rate: Learning rate.
    :param max_grad_norm: Gradient clipping norm.
    :param threshold: Threshold for the prediction confidence.
    :param custom_loss: Enable the custom loss (focal loss by default).
    :param weighted_loss: Enable the weighted bcewithlogits loss. Only works if the custom loss is enabled.
    :param fp16: Enable fp16 mixed precision training.
    :param eval_metric: Evaluation metric to use on the validation set.
            choices=[
            'loss', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples',
            'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples',
            'matthews_macro', 'matthews_micro',
            'roc_auc_micro', 'roc_auc_macro', 'roc_auc_weighted', 'roc_auc_samples',
            'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples',
            'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples',
            'hamming_loss', 'accuracy', 'ndcg_1', 'ndcg_3', 'ndcg_5', 'ndcg_10'],
            help="Evaluation metric to use on the validation set.")
    :param full_metrics: Compute all the metrics during the evaluation.
    :param trust_remote: Trust the remote code for the model.
    :param save_class_report: Save the classification report.
    :param class_report_step: Number of epochs before creating a new classification report.
    :return:
    """
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
            report_to="all",
            fp16=fp16,
        )

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

        # print(f"Best checkpoint path: {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    # fmt: off

    CLI(start_train)
