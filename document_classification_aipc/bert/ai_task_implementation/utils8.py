from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, hamming_loss, ndcg_score, precision_score, recall_score, jaccard_score, matthews_corrcoef, multilabel_confusion_matrix, zero_one_loss
from torch import sigmoid, Tensor, stack, from_numpy
import os
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.ops import focal_loss
from transformers import Trainer
from torch import nn, Tensor, nonzero, sort
import re
from transformers import AutoTokenizer
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from torch import tensor, ones_like
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
import gzip
import pickle
import math
from copy import deepcopy
from datetime import datetime
from pagerange import PageRange

import pickle
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
    
class CustomTrainer(Trainer):
    """
    Custom Trainer to compute the weighted BCE loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_weights = None
        self.use_focal_loss = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2

    def prepare_labels(self, data_path, language, seed, device):
        """
        Set the mlb encoder and the weights for the BCE loss.

        :param data_path: Path to the data.
        :param language: Language of the data.
        :param seed: Seed of the data.
        :param device: Device to use.
        """
        # Load the weights
        with open(os.path.join(data_path, language, f"split_{seed}", "train_labs_count.json"), "r") as weights_fp:
            data = json.load(weights_fp)
            weights = []
            """ # Approach with max weight in case of 0
            for key in data["labels"]:
                # Each weight is the inverse of the frequency of the label. Negative / positive
                weights.append((data["total_samples"] - data["labels"][key])/data["labels"][key] if data["labels"][key] != 0 else None)
            
            # If the weight is None, set it to the maximum weight
            max_weight = max([w for w in weights if w is not None])
            weights = [w if w else max_weight for w in weights] """

            for key in data["labels"]:
                # Each weight is the inverse of the frequency of the label. Negative / positive
                weights.append((data["total_samples"] - data["labels"][key] + 1e-10)/(data["labels"][key] + 1e-10))

            self.custom_weights = Tensor(weights).to(device)
        
    def set_weighted_loss(self):
        """
        Set the loss to the weighted BCE loss.
        """
        self.use_focal_loss = False

    def set_focal_params(self, alpha, gamma):
        """
        Set the focal loss parameters.

        :param alpha: Alpha parameter.
        :param gamma: Gamma parameter.
        """
        self.focal_alpha = alpha
        self.focal_gamma = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function to compute either the focal loss or the weighted BCE loss.

        :param model: Model to use.
        :param inputs: Inputs to the model.
        :param return_outputs: Whether to return the outputs. Default: False.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            logits = outputs.get("logits")
            
            if self.use_focal_loss:
                loss = focal_loss.sigmoid_focal_loss(
                    logits,
                    labels,
                    reduction="mean",
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma
                    )
            else:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.custom_weights)
                loss = loss_fct(logits, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    

def sklearn_metrics_core(y_true, predictions, data_path, threshold=0.5, get_conf_matrix=False, get_class_report=False):
    """
    Shared code for the sklearn metrics.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param data_path: Path to the data.
    :param threshold: Threshold to use for the predictions.
    :param get_conf_matrix: Whether to get the confusion matrix.
    :param get_class_report: Whether to get the classification report.
    :return: Initialized variables.
    """
    # Convert the predictions to binary
    probs = sigmoid(Tensor(predictions))
    y_pred = (probs.detach().numpy() >= threshold).astype(int)

    if get_conf_matrix:
        with open(os.path.join(data_path, 'mlb_encoder.pickle'), 'rb') as f:
            mlb_encoder = pickle.load(f)
        
        # labels = mlb_encoder.inverse_transform(np.ones((1, y_true.shape[1])))[0]
        labels = mlb_encoder.classes_.tolist()
        mlb_conf = multilabel_confusion_matrix(y_true, y_pred)
        conf_matrix = {}
        for i in range(len(labels)):
            conf_matrix[labels[i]] = mlb_conf[i].tolist()
    else:
        conf_matrix = None
    
    if get_class_report:
        class_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True, digits=4)
        class_report = {
            key: value for key, value in class_report.items() if key.isnumeric()# and value['support'] > 0
        }
    else:
        class_report = None

    to_return = {}

    return probs, y_pred, conf_matrix, class_report, to_return


def sklearn_metrics_full(y_true, predictions, data_path, threshold=0.5, get_conf_matrix=False, get_class_report=False, parent_handling="none"):
    """
    Return all the metrics and the classification report for the predictions.
    
    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions.
    :param get_conf_matrix: If True, return the confusion matrix.
    :param get_class_report: If True, return the classification report.
    :param parent_handling: How to handle the parent labels.
    :return: A dictionary with the metrics and a classification report.
    """
    probs, y_pred, conf_matrix, class_report, to_return = sklearn_metrics_core(y_true, predictions, data_path, threshold, get_conf_matrix, get_class_report)

    to_return = calculate_metrics(y_true, y_pred, probs, to_return)

    if parent_handling == "add" or parent_handling == "builtin":
        to_return.update(calculate_parent_metrics(y_true, predictions, data_path, parent_handling))

    return to_return, class_report, conf_matrix


def sklearn_metrics_single(y_true, predictions, data_path, threshold=0.5, get_conf_matrix=False, get_class_report=False,
    eval_metric=''):
    """
    Return the specified metric and the classification report for the predictions during the training.
    
    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions.
    :param get_conf_matrix: If True, return the confusion matrix.
    :param get_class_report: If True, return the classification report.
    :param eval_metric: The metric to use for the evaluation.
    :return: A dictionary with the metric and a classification report.
    """
    _, y_pred, conf_matrix, class_report, to_return = sklearn_metrics_core(y_true, predictions, data_path, threshold, get_conf_matrix, get_class_report)

    if "accuracy" in eval_metric:
        to_return["accuracy"] = accuracy_score(y_true, y_pred)
    elif "f1" in eval_metric:
        to_return[eval_metric] = f1_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "precision" in eval_metric:
        to_return[eval_metric] = precision_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "recall" in eval_metric:
        to_return[eval_metric] = recall_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "hamming" in eval_metric:
        to_return["hamming_loss"] = hamming_loss(y_true, y_pred)
    elif "jaccard" in eval_metric:
        to_return[eval_metric] = jaccard_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "matthews" in eval_metric:
        references = np.array(y_true)
        predictions = np.array(y_pred)
        if eval_metric == "matthews_micro":
            to_return["matthews_micro"] = matthews_corrcoef(y_true=references.ravel(), y_pred=predictions.ravel())
        elif eval_metric == "matthews_macro":
            to_return["matthews_macro"] = np.mean([
                matthews_corrcoef(y_true=references[:, i], y_pred=predictions[:, i], sample_weight=None)
                for i in range(references.shape[1])
            ])
    elif "roc_auc" in eval_metric:
        to_return[eval_metric] = roc_auc_score(y_true, y_pred, average=eval_metric.split("_")[1])
    elif "ndcg" in eval_metric:
        to_return[eval_metric] = ndcg_score(y_true, y_pred, k=eval_metric.split("_")[1])

    return to_return, class_report, conf_matrix


def calculate_metrics(y_true, y_pred, probs, to_return):
    """
    Calculates the metrics.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param probs: Predicted probabilities.
    :param to_return: Dictionary to return.
    """
    averaging=["micro", "macro", "weighted", "samples"]

    to_return["accuracy"] = accuracy_score(y_true, y_pred)
    
    true_labels = [nonzero(Tensor(labels), as_tuple=True)[0] for labels in y_true]
    pred_labels = sort(probs, descending=True)[1][:, :6]
    pk_scores = [np.intersect1d(true, pred).shape[0] / (pred.shape[0] + 1e-10) for true, pred in
                    zip(true_labels, pred_labels)]
    rk_scores = [np.intersect1d(true, pred).shape[0] / (true.shape[0] + 1e-10) for true, pred in
                    zip(true_labels, pred_labels)]
    f1k_scores = [2 * recall * precision / (recall + precision + 1e-10) for recall, precision in zip(pk_scores, rk_scores)]
    to_return["f1"] = sum(f1k_scores) / len(f1k_scores)
    for avg in averaging:
        to_return[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    for avg in averaging:
        to_return[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    
    for avg in averaging:
        to_return[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    
    to_return["hamming_loss"] = hamming_loss(y_true, y_pred)
    
    for avg in averaging:
        to_return[f"jaccard_{avg}"] = jaccard_score(y_true, y_pred, average=avg, zero_division=0)
    
    references = np.array(y_true)
    predictions = np.array(y_pred)
    to_return["matthews_macro"] = np.mean([
        matthews_corrcoef(y_true=references[:, i], y_pred=predictions[:, i], sample_weight=None)
        for i in range(references.shape[1])
    ])
    to_return["matthews_micro"] = matthews_corrcoef(y_true=references.ravel(), y_pred=predictions.ravel())
    
    for avg in averaging:
        try:
            to_return[f"roc_auc_{avg}"] = roc_auc_score(y_true, y_pred, average=avg)
        except ValueError:
            to_return[f"roc_auc_{avg}"] = 0.0
    
    for k in [1, 3, 5, 10]:
        to_return[f"ndcg_{k}"] = ndcg_score(y_true, y_pred, k=k)

    to_return["zero_one_loss"] = int(zero_one_loss(y_true, y_pred, normalize=False))

    to_return["total_samples"] = len(y_true)

    return to_return


def calculate_parent_metrics(y_true, predictions, data_path, mode):
    """
    Get the parent label metrics.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param data_path: Path to the data.
    :return: Metrics.
    """
    # Convert the predictions to binary
    probs = sigmoid(Tensor(predictions))
    y_pred = (probs.detach().numpy() >= 0.5).astype(int)

    # Get the labels
    if mode == "add":
        mt_labels_true, mt_labels_pred, do_labels_true, do_labels_pred = add_setup(data_path, y_true, y_pred)
    else:
        mt_labels_true, mt_labels_pred, do_labels_true, do_labels_pred = builtin_setup(data_path, y_true, y_pred)

    # create the lists to use to calculate the F1 score
    true_labels = [nonzero(Tensor(labels), as_tuple=True)[0] for labels in y_true]
    (mt_labels_true_manual, mt_labels_pred_manual,
     do_labels_true_manual, do_labels_pred_manual) = initialize_manual_labels(true_labels, probs)
    
    # convert the dictionaries to lists with only the values
    mt_labels_true = [list(mt_labels_true[i].values()) for i in range(len(mt_labels_true))]
    mt_labels_pred = [list(mt_labels_pred[i].values()) for i in range(len(mt_labels_pred))]
    do_labels_true = [list(do_labels_true[i].values()) for i in range(len(do_labels_true))]
    do_labels_pred = [list(do_labels_pred[i].values()) for i in range(len(do_labels_pred))]
        
    metrics = {}

    for label_type in ["mt", "do"]:
        labels_true = mt_labels_true if label_type == "mt" else do_labels_true
        labels_pred = mt_labels_pred if label_type == "mt" else do_labels_pred

        new_metrics = calculate_metrics(labels_true, labels_pred, probs, {})

        # Calculate the F1 score
        # It needs to be redone here because the labels are in simple arrays while in 'calculate_metrics' it uses tensors
        labels_true = mt_labels_true_manual if label_type == "mt" else do_labels_true_manual
        labels_pred = mt_labels_pred_manual if label_type == "mt" else do_labels_pred_manual
        pk_scores = []
        rk_scores = []
        for true, pred in zip(labels_true, labels_pred):
            pk_scores.append(np.intersect1d(true, pred).shape[0] / (len(pred) + 1e-10))
            rk_scores.append(np.intersect1d(true, pred).shape[0] / (len(true) + 1e-10))
        f1k_scores = [2 * recall * precision / (recall + precision + 1e-10) for recall, precision in zip(pk_scores, rk_scores)]
        new_metrics["f1"] = sum(f1k_scores) / len(f1k_scores)

        keys = [key + f"_{label_type}" for key in list(new_metrics)]

        to_update = {key: new_metrics[key.replace(f"_{label_type}", "")] for key in keys}
        
        metrics.update(to_update)

    return metrics

def initialize_manual_labels(true_labels, probs):
    """
    Initialize the arrays of labels to be to calculate the F1 for the parent labels.

    :param true_labels: Array with the true labels in the format:
        [
            ["3254", "4567", "2728", ...],
            ...
        ]
    :param probs: Predicted probabilities.
    :return: Arrays with the true and predicted MT and DO labels in the format:
        [
            ["3254", "4567", "2728", ...],
            ...
        ]
    """
    with open("./config/mt_labels.json", "r") as fp:
        mt_mapping = json.load(fp)

    pred_labels = sort(probs, descending=True)[1][:, :5]

    true_labels_mt = []
    true_labels_do = []
    for labels in true_labels:
        # Every label should be present in the mapping, only the label "eurovoc",
        # identified by id "3712", is not present and has no MT mapping.
        # See https://op.europa.eu/en/web/eu-vocabularies/concept/-/resource?uri=http://eurovoc.europa.eu/3712
        to_append_mt = []
        to_append_do = []
        for label in labels:
            if str(label.item()) in mt_mapping:
                to_append_mt.append(mt_mapping[str(label.item())])
                to_append_do.append(mt_mapping[str(label.item())][:2])
        true_labels_mt.append(to_append_mt)
        true_labels_do.append(to_append_do)
    
    pred_labels_mt = []
    pred_labels_do = []
    for labels in pred_labels:
        to_append_mt = []
        to_append_do = []
        for label in labels:
            if str(label.item()) in mt_mapping:
                to_append_mt.append(mt_mapping[str(label.item())])
                to_append_do.append(mt_mapping[str(label.item())][:2])
        pred_labels_mt.append(to_append_mt)
        pred_labels_do.append(to_append_do)

    return true_labels_mt, pred_labels_mt, true_labels_do, pred_labels_do

def add_setup(data_path, y_true, y_pred):
    """
    Initialize the parent labels by adding the parents artificially
    (the Thesaurus Concept labels are mapped to their parent Micro Thesaurus and Domain labels).

    :param data_path: Path to the data.
    :param y_true: True labels.
    :param y_pred: Predictions.
    :return: Labels.
    """
    with open(os.path.join(data_path, 'mlb_encoder.pickle'), 'rb') as f:
        mlb_encoder = pickle.load(f)
        
        labels_true = mlb_encoder.inverse_transform(y_true)
        labels_pred = mlb_encoder.inverse_transform(y_pred)

        # load the file "mt_position" from the config folder and the mapping
        with open("./config/mt_labels_position.json", "r") as fp:
            mt_position = json.load(fp)
        with open("./config/mt_labels.json", "r") as fp:
            mt_mapping = json.load(fp)
        # initialize a dictionary for both true and pred labels
        mt_labels_true = [{k:0 for k in mt_position} for _ in range(len(labels_true))]
        mt_labels_pred = [{k:0 for k in mt_position} for _ in range(len(labels_pred))]
        # if the label is present, set the value to 1
        for i in range(len(labels_true)):
            for label in labels_true[i]:
                if label in mt_mapping:
                    mt_labels_true[i][mt_mapping[label]] = 1
        for i in range(len(labels_pred)):
            for label in labels_pred[i]:
                if label in mt_mapping:
                    mt_labels_pred[i][mt_mapping[label]] = 1
        # load the file "do_labels_position" from the config folder
        with open("./config/domain_labels_position.json", "r") as fp:
            do_position = json.load(fp)
        # initialize a dictionary for both true and pred labels
        do_labels_true = [{k:0 for k in do_position} for _ in range(len(labels_true))]
        do_labels_pred = [{k:0 for k in do_position} for _ in range(len(labels_pred))]
        # if the label is present, set the value to 1
        for i in range(len(mt_labels_true)):
            for label in mt_labels_true[i]:
                if mt_labels_true[i][label] == 1:
                    do_labels_true[i][label[:2]] = 1
        for i in range(len(mt_labels_pred)):
            for label in mt_labels_pred[i]:
                if mt_labels_pred[i][label] == 1:
                    do_labels_pred[i][label[:2]] = 1
    
    return mt_labels_true, mt_labels_pred, do_labels_true, do_labels_pred

def builtin_setup(data_path, y_true, y_pred):
    """
    Initialize the parent labels if the parents are already present in the training data
    (only if the data for the model was processed with the --add_mt_do flag).

    :param data_path: Path to the data.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Labels.
    """
    with open(os.path.join(data_path, 'mlb_encoder.pickle'), 'rb') as f:
        mlb_encoder = pickle.load(f)
        
        labels_true = mlb_encoder.inverse_transform(y_true)
        labels_pred = mlb_encoder.inverse_transform(y_pred)

        # load the file "mt_position" from the config folder and the mapping
        with open("./config/mt_labels_position.json", "r") as fp:
            mt_position = json.load(fp)
        # load the file "domain_labels_position" from the config folder
        with open("./config/domain_labels_position.json", "r") as fp:
            do_position = json.load(fp)

        # initialize a dictionary for both true and pred labels
        mt_labels_true = [{k:0 for k in mt_position} for _ in range(len(labels_true))]
        mt_labels_pred = [{k:0 for k in mt_position} for _ in range(len(labels_pred))]
        do_labels_true = [{k:0 for k in do_position} for _ in range(len(labels_true))]
        do_labels_pred = [{k:0 for k in do_position} for _ in range(len(labels_pred))]

        # if the label is present, set the value to 1
        for i in range(len(labels_true)):
            for label in labels_true[i]:
                if "_mt" in label:
                    mt_labels_true[i][label.split("_mt")[0]] = 1
                elif "_do" in label:
                    do_labels_true[i][label.split("_do")[0]] = 1
        for i in range(len(labels_pred)):
            for label in labels_pred[i]:
                if "_mt" in label:
                    mt_labels_pred[i][label.split("_mt")[0]] = 1
                elif "_do" in label:
                    do_labels_pred[i][label.split("_do")[0]] = 1
    
    return mt_labels_true, mt_labels_pred, do_labels_true, do_labels_pred

def data_collator_tensordataset(features):
    """
    Custom data collator for datasets of the type TensorDataset.

    :param features: List of features.
    :return: Batch.
    """
    batch = {}
    batch['input_ids'] = stack([f[0] for f in features])
    batch['attention_mask'] = stack([f[1] for f in features])
    batch['labels'] = stack([f[2] for f in features])
    
    return batch


def load_data(data_path, lang, data_type, seed):
    """
    Load the data from the specified directory.

    :param data_path: Path to the data.
    :param lang: Language.
    :param data_type: Type of data to load (train or test).
    :param seed: Seed to load.
    :return: List of train, dev and test loaders.
    """
    to_return = []

    for directory in os.listdir(data_path):
        if lang == directory:
            if data_type == "train":
                print("\nLoading training and dev data from directory {}...".format(os.path.join(data_path, directory, f"split_{seed}")))

                # The data is stored in numpy arrays, so it has to be converted to tensors.
                train_X = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "train_X.npy")))
                train_mask = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "train_mask.npy")))
                train_y = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "train_y.npy"))).float()

                assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]

                dev_X = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "dev_X.npy")))
                dev_mask = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "dev_mask.npy")))
                dev_y = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "dev_y.npy"))).float()

                assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]

                dataset_train = TensorDataset(train_X, train_mask, train_y)

                dataset_dev = TensorDataset(dev_X, dev_mask, dev_y)
                to_return = [dataset_train, dataset_dev, train_y.shape[1]]

            elif data_type == "test":
                print("\nLoading test data from directory {}...".format(os.path.join(data_path, directory, f"split_{seed}")))
                test_X = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "test_X.npy")))
                test_mask = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "test_mask.npy")))
                test_y = from_numpy(np.load(os.path.join(data_path, directory, f"split_{seed}", "test_y.npy"))).float()

                assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]

                dataset_test = TensorDataset(test_X, test_mask, test_y)

                to_return = dataset_test
            break

    return to_return


def save_splits(X, masks, y, directory, mlb, data_path, seeds):
    """
    Save the splits of the dataset.
    
    :param X: List of inputs.
    :param masks: List of masks.
    :param y: List of labels.
    :param directory: Language directory.
    :param mlb: MultiLabelBinarizer object.
    """

    print(f"{datetime.now().replace(microsecond=0)} - Saving splits...")

    for seed in seeds:
        np.random.seed(int(seed))
        # Create two splits:test+dev and train
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 0.8])
        train_idx, aux_idx = next(stratifier.split(X, y))
        train_X, train_mask, train_y = X[train_idx, :], masks[train_idx, :], y[train_idx, :]

        assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]

        # Create two splits: test and dev
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.5, 0.5])
        dev_idx, test_idx = next(stratifier.split(X[aux_idx, :], y[aux_idx, :]))
        dev_X, dev_mask, dev_y = X[aux_idx, :][dev_idx, :], masks[aux_idx, :][dev_idx, :], y[aux_idx, :][dev_idx, :]
        test_X, test_mask, test_y = X[aux_idx, :][test_idx, :], masks[aux_idx, :][test_idx, :], y[aux_idx, :][test_idx, :]

        assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]
        assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]

        to_print = f"{seed} - Splitted the documents in - train: {train_X.shape[0]}, dev: {dev_X.shape[0]}, test: {test_X.shape[0]}"
        print(to_print)

        with open(os.path.join(data_path, directory, "stats.txt"), "a+") as f:
            f.write(to_print + "\n")

        if not os.path.exists(os.path.join(data_path, directory, f"split_{seed}")):
            os.makedirs(os.path.join(data_path, directory, f"split_{seed}"))

        # Save the splits
        np.save(os.path.join(data_path, directory, f"split_{seed}", "train_X.npy"), train_X)
        np.save(os.path.join(data_path, directory, f"split_{seed}", "train_mask.npy"), train_mask)
        np.save(os.path.join(data_path, directory, f"split_{seed}", "train_y.npy"), train_y)

        np.save(os.path.join(data_path, directory, f"split_{seed}", "dev_X.npy"), dev_X)
        np.save(os.path.join(data_path, directory, f"split_{seed}", "dev_mask.npy"), dev_mask)
        np.save(os.path.join(data_path, directory, f"split_{seed}", "dev_y.npy"), dev_y)

        np.save(os.path.join(data_path, directory, f"split_{seed}", "test_X.npy"), test_X)
        np.save(os.path.join(data_path, directory, f"split_{seed}", "test_mask.npy"), test_mask)
        np.save(os.path.join(data_path, directory, f"split_{seed}", "test_y.npy"), test_y)

        # Save the counts of each label, useful for weighted loss
        sample_labs = mlb.inverse_transform(train_y)
        labs_count = {"total_samples": len(sample_labs), "labels": {label: 0 for label in mlb.classes_}}

        for sample in sample_labs:
            for label in sample:
                labs_count["labels"][label] += 1
        
        with open(os.path.join(data_path, directory, f"split_{seed}", "train_labs_count.json"), "w") as fp:
            json.dump(labs_count, fp)

        # Shuffle the splits using the random seed for reproducibility
        X, masks, y = shuffle(X, masks, y, random_state=int(seed))

def process_year(path, tokenizer_name, max_length, limit_tokenizer, get_doc_ids, add_mt_do, title_only, add_title, summarized):
    """
    Process a year of the dataset.

    :param path: Path to the data.
    :param tokenizer_name: Name of the tokenizer to use.
    :param args: Command line arguments.
    :return: List of inputs, masks and labels.
    """

    document_ct = 0
    big_document_ct = 0
    unk_ct = 0
    tokens_ct = 0

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenizer_kwargs = {"padding": "max_length", "truncation": True, "max_length": max_length}

    list_inputs = []
    list_labels = []
    list_masks = []

    with gzip.open(path, "rt", encoding="utf-8") as file:
        data = json.load(file)
        j = 1
        inputs_ids = tensor(tokenizer.encode(""))
        labels = list()
        if get_doc_ids:
            # Only get the document ids, without processing the text. Useful to know which documents go in which split.
            for doc in data:
                print(f"{datetime.now().replace(microsecond=0)} - {j}/{len(data)}", end="\r")
                j += 1
                labels = data[doc]["eurovoc_classifiers"]

                inputs_ids = tensor(tokenizer.encode(doc, **tokenizer_kwargs))

                list_inputs.append(inputs_ids)
                list_labels.append(labels)
                list_masks.append(ones_like(inputs_ids))
        else:
            # Process the text
            for doc in data:
                print(f"{datetime.now().replace(microsecond=0)} - {j}/{len(data)}", end="\r")
                j += 1
                text = ""
                if add_mt_do:
                    # Add MT and DO labels
                    labels = set(data[doc]["eurovoc_classifiers"]) if "eurovoc_classifiers" in data[doc] else set(data[doc]["eurovoc"])
                    to_add = set()
                    # The domains and microthesaurus labels are loaded from the json files
                    with open("config/domain_labels_position.json", "r") as fp:
                        domain = json.load(fp)
                    with open("config/mt_labels_position.json", "r") as fp:
                        microthesaurus = json.load(fp)
                    with open("config/mt_labels.json", "r", encoding="utf-8") as file:
                        mt_labels = json.load(file)
                    for label in labels:
                        if label in mt_labels:
                            if mt_labels[label] in microthesaurus:
                                to_add.add(mt_labels[label] + "_mt")
                            if mt_labels[label][:2] in domain:
                                to_add.add(mt_labels[label][:2] + "_do")
                    
                    labels = list(labels.union(to_add))
                else:
                    labels = data[doc]["eurovoc_classifiers"] if "eurovoc_classifiers" in data[doc] else data[doc]["eurovoc"]

                if title_only:
                    text = data[doc]["title"]
                else:
                    if add_title:
                        text = data[doc]["title"] + " "
                    
                    if summarized:
                        full_text = data[doc]["full_text"]
                        phrase_importance = []
                        i = 0

                        for imp in data[doc]["importance"]:
                            if not math.isnan(imp):
                                phrase_importance.append((i, imp))
                            i += 1
                        
                        phrase_importance = sorted(phrase_importance, key=lambda x: x[1], reverse=True)

                        # First, we get the most important phrases until the maximum length is reached.
                        if len(" ".join([full_text[phrase[0]] for phrase in phrase_importance]).split()) > max_length:
                            backup = deepcopy(phrase_importance)
                            while len(" ".join([full_text[phrase[0]] for phrase in phrase_importance]).split()) > max_length:
                                phrase_importance = phrase_importance[:-1]
                            phrase_importance.append(backup[len(phrase_importance)])

                        # Then, we sort the phrases by their position in the document.
                        phrase_importance = sorted(phrase_importance, key=lambda x: x[0])
                        text += " ".join([full_text[phrase[0]] for phrase in phrase_importance])
                    else:
                        text += data[doc]["full_text"] if "full_text" in data[doc] else data[doc]["text"]
                
                text = re.sub(r'\r', '', text)
                
                if limit_tokenizer:
                    # Here, the text is cut to the maximum length before being tokenized,
                    # potentially speeding up the process for long documents.
                    inputs_ids = tensor(tokenizer.encode(text, **tokenizer_kwargs))
                else:
                    inputs_ids = tensor(tokenizer.encode(text))

                if not limit_tokenizer:
                    document_ct += 1

                    # We count the number of unknown tokens and the total number of tokens.
                    for token in inputs_ids[1: -1]:
                        if token == tokenizer.unk_token_id:
                            unk_ct += 1

                        tokens_ct += 1

                    # If the input is over the maximum length, we cut it and increment the count of big documents.
                    if len(inputs_ids) > max_length:
                        big_document_ct += 1
                        inputs_ids = inputs_ids[:max_length]

                list_inputs.append(inputs_ids)
                list_labels.append(labels)
                list_masks.append(ones_like(inputs_ids))
    
    del data, inputs_ids, labels, tokenizer

    # Just some stats to print and save later.
    if len(list_inputs) == 0:
        print("No documents found in the dataset.")
        to_print = ""
    else:
        if not limit_tokenizer and not get_doc_ids:
            to_print = f"Dataset stats: - total documents: {document_ct}, big documents: {big_document_ct}, ratio: {big_document_ct / document_ct * 100:.4f}%"
            to_print += f"\n               - total tokens: {tokens_ct}, unk tokens: {unk_ct}, ratio: {unk_ct / tokens_ct * 100:.4f}%"
            print(to_print)
        else:
            to_print = ""

    return list_inputs, list_masks, list_labels, to_print

def process_datasets(data_path, directory, tokenizer_name, years, max_length, limit_tokenizer, get_doc_ids, add_mt_do, title_only, add_title, seeds):
    """
    Process the datasets and save them in the specified directory.

    :param data_path: Path to the data.
    :param directory: Language directory.
    :param tokenizer_name: Name of the tokenizer to use.
    """

    list_inputs = []
    list_masks = []
    list_labels = []
    list_stats = []
    list_years = []

    # If no years are specified, process all the downloaded years depending on the arguments.
    summarized = False
    if years == "all":
        years = [year for year in os.listdir(os.path.join(data_path, directory))
                      if os.path.isfile(os.path.join(data_path, directory, year))
                      and year.endswith(".json.gz")]
    else:
        years = PageRange(years).pages
        files_in_directory = [file for file in os.listdir(os.path.join(data_path, directory))
                              if file.endswith(".json.gz")]

        are_any_summarized = ["sum" in file for file in files_in_directory]
        if any(are_any_summarized):
            sum_type = files_in_directory[are_any_summarized.index(True)].split("_", 1)[1]
            years = [str(year) + f"_{sum_type}" for year in years]
        else:
            years = [str(year) + ".json.gz" for year in years]
        
    years = sorted(years)
    
    # Test if the file is summarized or not
    with gzip.open(os.path.join(data_path, directory, years[0]), "rt", encoding="utf-8") as file:
        data = json.load(file)
        if "importance" in data[tuple(data.keys())[0]]:
            summarized = True
        del data

    print(f"Files to process : {', '.join(years)}\n")

    for year in years:
        print(f"Processing file: '{year}'...")
        year_inputs, year_masks, year_labels, year_stats = process_year(os.path.join(data_path, directory, year), tokenizer_name, max_length, limit_tokenizer, get_doc_ids, add_mt_do, title_only, add_title, summarized)
        
        list_inputs += year_inputs
        list_masks += year_masks
        list_labels += year_labels
        list_stats.append(year_stats)
        list_years.append(year)

    assert len(list_inputs) == len(list_masks) == len(list_labels)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(list_labels)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    X = pad_sequence(list_inputs, batch_first=True, padding_value=tokenizer.pad_token_id).numpy()
    masks = pad_sequence(list_masks, batch_first=True, padding_value=0).numpy()

    # Save the MultiLabelBinarizer.
    with open(os.path.join(data_path, directory, "mlb_encoder.pickle"), "wb") as pickle_fp:
        pickle.dump(mlb, pickle_fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    if not limit_tokenizer:
        with open(os.path.join(data_path, directory, "stats.txt"), "w") as stats_fp:
            for year, year_stats in zip(list_years, list_stats):
                stats_fp.write(f"Year: {year}\n{year_stats}\n\n")

    save_splits(X, masks, y, directory, mlb, data_path, seeds)
