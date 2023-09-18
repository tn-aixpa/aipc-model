import argparse

parser = argparse.ArgumentParser(description='Save file for FastText.')
parser.add_argument("input_tint_folder", metavar="input-tint-folder", help="Folder with JSON files parsed by Tint")
parser.add_argument("input_file", metavar="input-file", help="JSON file with labels and IDs")
parser.add_argument("output_folder", metavar="output-folder", help="Output folder")
parser.add_argument("--test-ratio", help="Percentage of test data (default: 0.2)", type=float, default=0.2)
parser.add_argument("--dev-ratio", help="Percentage of dev data (default: 0.2)", type=float, default=0.2)

args = parser.parse_args()

import os
import json
import tqdm
import logging
import pickle
import fasttext
import pandas as pd
import numpy as np
import random
import scipy
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

testListName = os.path.join(args.output_folder, "testlist.txt")
devListName = os.path.join(args.output_folder, "devlist.txt")
testRatio = args.test_ratio
devRatio = args.dev_ratio

if testRatio + devRatio >= 1:
    log.error("dev + test must be less than 1")
    exit()

if testRatio + devRatio >= 0.8:
    log.warning("dev + test seems too big")

log.info("Loading JSON file")
with open(args.input_file, "r") as f:
    data = json.load(f)

trainSize = 0
testSize = 0
devSize = 0
testList = set()
devList = set()

classIndexes = {}
testSizes = {}
devSizes = {}
isMultiLabel = False

if isinstance(data, dict):
    data_tmp = []
    for record in data:
        data_tmp.append(data[record])
    data = data_tmp

for record in data:

    # Better to check this here, too
    tintFile = os.path.join(args.input_tint_folder, record['id'] + ".json")
    if not os.path.exists(tintFile):
        log.warning("File %s does not exist" % tintFile)
        continue

    if len(record['labels']) > 1:
        isMultiLabel = True

    for label in record['labels']:
        if label not in classIndexes:
            classIndexes[label] = set()
        classIndexes[label].add(record['id'])

if isMultiLabel:
    data_tmp = {}
    for record in data:
        data_tmp[record['id']] = record
    data = data_tmp

    for label in classIndexes:
        testSizes[label] = 0
        devSizes[label] = 0

    classIndexes = {k: v for k, v in sorted(classIndexes.items(), key=lambda item: len(item[1]))}

    for k in classIndexes:
        # print(classIndexes[k])
        # print(testSizes[k])
        tSize = math.ceil(testRatio * len(classIndexes[k])) - testSizes[k]
        if tSize > 0:
            pickSet = classIndexes[k].difference(testList).difference(devList)
            # choices = random.sample(pickSet, tSize)

            pickValues = {}
            for p in pickSet:
                pickValues[p] = 0
                for l in data[p]['labels']:
                    if testSizes[l] > 0:
                        pickValues[p] += 1
            pickValues = {k: v for k, v in sorted(pickValues.items(), key=lambda item: item[1])}

            choices = [*pickValues.keys()][0:tSize]
            for c in choices:
                testList.add(c)
                for l in data[c]['labels']:
                    testSizes[l] += 1

        dSize = math.ceil(devRatio * len(classIndexes[k])) - devSizes[k]
        if dSize > 0:
            pickSet = classIndexes[k].difference(devList).difference(testList)
            # choices = random.sample(pickSet, tSize)

            pickValues = {}
            for p in pickSet:
                pickValues[p] = 0
                for l in data[p]['labels']:
                    if devSizes[l] > 0:
                        pickValues[p] += 1
            pickValues = {k: v for k, v in sorted(pickValues.items(), key=lambda item: item[1])}

            choices = [*pickValues.keys()][0:dSize]
            for c in choices:
                devList.add(c)
                for l in data[c]['labels']:
                    devSizes[l] += 1


    testSize = len(testList)
    devSize = len(devList)
    trainSize = len(data) - testSize - devSize

    for k in classIndexes:
        dtRatio = devRatio + testRatio
        thisRatio = (testSizes[k] + devSizes[k]) / len(classIndexes[k])
        if thisRatio - dtRatio > 0.2:
            log.warning("Unbalanced training ratio for class %s (total: %d): %.2f - test: %d - dev: %d - train: %d" %
                (k, len(classIndexes[k]), thisRatio, testSizes[k], devSizes[k], len(classIndexes[k]) - testSizes[k] - devSizes[k]))

else:
    ### Not tested

    for label in classIndexes:
        testSizes[label] = 0
        devSizes[label] = 0
        num = len(classIndexes[label])
        if num > 1:
            testSizes[label] = math.ceil(testRatio * num)
            devSizes[label] = math.ceil(devRatio * num)

    log.info("Shuffling data")
    random.shuffle(data)

    log.info("Extracting texts")
    for record in data:
        tintFile = os.path.join(args.input_tint_folder, record['id'] + ".json")
        if not os.path.exists(tintFile):
            log.warning("File %s does not exist" % tintFile)
            continue

        if len(record['labels']) == 0:
            log.warning("File %s has no labels, skipping" % tintFile)
            continue

        isTest = False
        isDev = False
        if testSizes[record['labels'][0]] > 0:
            testSizes[record['labels'][0]] -= 1
            isTest = True
        if not isTest and devSizes[record['labels'][0]] > 0:
            devSizes[record['labels'][0]] -= 1
            isDev = True

        if isTest:
            testList.add(record['id'])
            testSize += 1
        elif isDev:
            devList.add(record['id'])
            devSize += 1
        else:
            trainSize += 1

log.info("Saving test list")
with open(testListName, "w") as fw:
    for docID in testList:
        fw.write(docID)
        fw.write("\n")

log.info("Saving dev list")
with open(devListName, "w") as fw:
    for docID in devList:
        fw.write(docID)
        fw.write("\n")

log.info("Train size: %d" % trainSize)
log.info("Test size: %d" % testSize)
log.info("Dev size: %d" % devSize)
