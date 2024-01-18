import argparse

parser = argparse.ArgumentParser(description='Save file for FastText.')
parser.add_argument("input_tint_folder", metavar="input-tint-folder", help="Folder with JSON files parsed by Tint")
parser.add_argument("input_file", metavar="input-file", help="JSON file with labels and IDs")
parser.add_argument("output_folder", metavar="output-folder", help="Output folder")

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

testListName = os.path.join(args.output_folder, "testlist.txt")
devListName = os.path.join(args.output_folder, "devlist.txt")

completeFileName = os.path.join(args.output_folder, "complete.json")

def getTintInfo(data):
    goodTokens = []
    allLemmas = []
    allTokens = []
    for sentence in data['sentences']:
        for token in sentence['tokens']:
            pos = token['pos']
            if token['originalText'].startswith("."):
                continue
            if token['originalText'].endswith("\x92"):
                continue

            allLemmas.append(token['lemma'].lower())
            if (not token['isMultiwordToken']) or token['isMultiwordFirstToken']:
                allTokens.append(token['originalText'].replace(" ", "_").lower())

            if pos == "SP":
                continue
            if len(token['originalText']) < 3:
                continue
            if pos.startswith("A") or pos.startswith("S") or pos.startswith("V"):
                goodTokens.append(token['lemma'].lower())
    return goodTokens, allLemmas, allTokens

log.info("Loading JSON file")
with open(args.input_file, "r") as f:
    data = json.load(f)

if isinstance(data, dict):
    data_tmp = []
    for record in data:
        data_tmp.append(data[record])
    data = data_tmp

if not os.path.exists(testListName):
    log.error("No test file found")
    exit()

testList = []
devList = []
with open(testListName, "r") as f:
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        testList.append(line)

if os.path.exists(devListName):
    with open(devListName, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            devList.append(line)


log.info("Extracting texts")
textOnlyCorpus = []
for record in tqdm.tqdm(data):
    tintFile = os.path.join(args.input_tint_folder, record['id'] + ".json")
    if not os.path.exists(tintFile):
        log.warning("File %s does not exist" % tintFile)
        continue

    if len(record['labels']) == 0:
        log.warning("File %s has no labels, skipping" % tintFile)
        continue

    with open(tintFile, "r") as f:
        tintData = json.load(f)
    goodTokens, allLemmas, allTokens = getTintInfo(tintData)

    record['goodTokens'] = goodTokens
    record['allLemmas'] = allLemmas
    record['allTokens'] = allTokens

    record['test'] = 0
    record['dev'] = 0
    if record['id'] in testList:
        record['test'] = 1
    elif record['id'] in devList:
        record['dev'] = 1

log.info("Saving file")
with open(completeFileName, 'w') as fw:
    fw.write(json.dumps(data, indent = 4))
