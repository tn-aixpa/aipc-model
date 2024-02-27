import logging
import os
import zipfile
import numpy as np
import operator
import collections
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
import mlrun
from mlrun import MLClientCtx

# 06-micro_macro.py refactored into callable functions

def evaluate(context: MLClientCtx, pred_files: mlrun.DataItem, gold_files: mlrun.DataItem, show_cm: bool = False, probThreshold: float = 0.5):
    """
    Get information on the performances of the trained models. Accepts:
    - pred_files: zip with predictions files, in FastText format
    - gold_files: zip with test files, in FastText format
    - probThreshold: probability threshold below which the prediction is discarded (default: 0.5)
    - show_cm: show confusion matrix
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("Downloading and extracting pred_files")
    pred_files.download(target_path="results.zip")

    with zipfile.ZipFile("results.zip", "r") as z:
        z.extractall() #extracts a folder named results

    pred_files_folder = "results"

    logging.info("Downloading and extracting gold_files")
    gold_files.download(target_path="filtering_files.zip")

    with zipfile.ZipFile("filtering_files.zip", "r") as z:
        z.extractall() #extracts a folder named filtering_files

    gold_files_folder = "filtering_files"

    for file in os.listdir(pred_files_folder):
        y_true = []
        y_pred = []

        pred_file = os.path.join(pred_files_folder, file)
        gold_file = os.path.join(gold_files_folder, file.split(".")[0] + ".test.txt")
        logging.info(f"Pred file: {pred_file}")
        logging.info(f"Test file: {gold_file}")

    # for key, uri in pred_files.items():
        # y_true = []
        # y_pred = []

        # pred_file = key + ".txt"
        # logging.info(f"Downloading {pred_file}")
        # mlrun.get_dataitem(uri).download(target_path=pred_file)

        # gold_file_key = key.split(".")[0] + ".test"
        # gold_file = gold_file_key + ".txt"
        # logging.info(f"Downloading {gold_file}")
        # mlrun.get_dataitem(gold_files[gold_file_key]).download(target_path=gold_file)

        with open(pred_file) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                parts = line.split()
                y = []
                hasProb = False
                if len(parts) > 1 and not parts[1].startswith("__label__"):
                    hasProb = True

                if hasProb:
                    for i in range(1, len(parts), 2):
                        if len(y) == 0:
                            y.append(parts[i - 1])
                        else:
                            if float(parts[i]) > probThreshold:
                                y.append(parts[i - 1])
                else:
                    for part in parts:
                        if part.startswith("__label__"):
                            y.append(part)
                y_pred.append(y)

        with open(gold_file) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                parts = line.split()
                y = []
                for part in parts:
                    if part.startswith("__label__"):
                        y.append(part)
                y_true.append(y)

        m = MultiLabelBinarizer().fit(y_true)

        macro_f1_score = f1_score(m.transform(y_true), m.transform(y_pred), average='macro')
        micro_f1_score = f1_score(m.transform(y_true), m.transform(y_pred), average='micro')
        weighted_f1_score = f1_score(m.transform(y_true), m.transform(y_pred), average='weighted')

        context.log_result(f"{file.split('.')[0]}_macro", macro_f1_score)
        context.log_result(f"{file.split('.')[0]}_micro", micro_f1_score)
        context.log_result(f"{file.split('.')[0]}_weighted", weighted_f1_score)

        if show_cm:
            y_true = [x[0] for x in y_true]
            y_pred = [x[0] for x in y_pred]

            distribution = {i:y_true.count(i) for i in set(y_true)}
            sorted_x = sorted(distribution.items(), key=operator.itemgetter(1), reverse=True)
            distribution = collections.OrderedDict(sorted_x)

            labels = set()
            for l in y_true:
                labels.add(l)
            for l in y_pred:
                labels.add(l)
            labels = list(labels)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            errors_dict = {}
            errors_dict_single = {}
            errors = {}
            rows, cols = np.nonzero(cm)
            for i in range(len(rows)):
                if rows[i] == cols[i]:
                    continue
                v = cm[rows[i]][cols[i]]
                if v == 1:
                    continue
                l1 = labels[rows[i]]
                l2 = labels[cols[i]]
                if l1 not in errors_dict_single:
                    errors_dict_single[l1] = 0
                if l1 not in errors:
                    errors[l1] = {}
                errors[l1][l2] = v
                errors_dict_single[l1] += v
                errors_dict[l1 + " " + l2] = v
                # print(l1, l2, v)
            sorted_x = sorted(errors_dict.items(), key=operator.itemgetter(1), reverse=True)
            errors_dict = collections.OrderedDict(sorted_x)
            sorted_x = sorted(errors_dict_single.items(), key=operator.itemgetter(1), reverse=True)
            errors_dict_single = collections.OrderedDict(sorted_x)
            errors_dict_ratio = {}
            for k in errors_dict_single:
                errors_dict_ratio[k] = errors_dict_single[k] / distribution[k]
            sorted_x = sorted(errors_dict_ratio.items(), key=operator.itemgetter(1), reverse=True)
            errors_dict_ratio = collections.OrderedDict(sorted_x)
            # print(errors_dict)
            # print(errors_dict_single)
            # print(errors_dict_ratio)
            # print(errors)

            cm_values = {}

            for k in list(errors_dict_single.keys())[:5]:
                cm_values[str(k)] = {
                    "Total": distribution[k],
                    "Errors": errors_dict_single[k],
                    "Ratio": errors_dict_ratio[k],
                    "Errors_matrix": errors[k]
                }

            context.log_result(f"{file.split('.')[0]}_confusion_matrix", cm_values)
