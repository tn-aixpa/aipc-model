import argparse

parser = argparse.ArgumentParser(description='Get micro/macro.')
parser.add_argument("pred_file", metavar="pred-file", help="Predictions file")
parser.add_argument("gold_file", metavar="gold-file", help="Gold file")
parser.add_argument("--prob-threshold", help="Probability threshold (default: 0.5)", type=float, default=0.5)
parser.add_argument("--show-cm", help="Show confusion matrix", action='store_true')

args = parser.parse_args()

import logging
import numpy as np
import operator
import collections

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix

probThreshold = args.prob_threshold

logging.basicConfig(level=logging.INFO)

y_true = []
y_pred = []

with open(args.pred_file) as f:
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

with open(args.gold_file) as f:
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

print("Macro:", macro_f1_score)
print("Micro:", micro_f1_score)
print("Weighted:", weighted_f1_score)

if args.show_cm:
    print()
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

    for k in list(errors_dict_single.keys())[:5]:
        print("### " + str(k))
        print("Total: " + str(distribution[k]))
        print("Errors: " + str(errors_dict_single[k]))
        print("Ratio: " + str(errors_dict_ratio[k]))
        print("Errors matrix: " + str(errors[k]))
        print()
