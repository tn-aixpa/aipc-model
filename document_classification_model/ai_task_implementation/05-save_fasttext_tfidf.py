import argparse

parser = argparse.ArgumentParser(description='Save file for FastText.')
parser.add_argument("folder", help="Folder with complete.json file")
parser.add_argument("--min-freq", help="Min frequency for stopwords (default: 5)", type=int, default=5)
# parser.add_argument("--min-weight", help="Min weight for TF-IDF (default: 2.0)", type=float, default=2.0)

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
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer

types = ['goodTokens', 'allLemmas', 'allTokens']
# types = ['goodTokens']

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

completeFileName = os.path.join(args.folder, "complete.json")
if not os.path.exists(completeFileName):
    log.error("No JSON file found")
    exit()

minFreq = args.min_freq

minWeight = {}
minWeight['by_document'] = 2.0
minWeight['by_label'] = 1.0

def intersection(lst1, lst2):
    lst2set = set(lst2)
    lst3 = [value for value in lst1 if value in lst2set]
    return lst3

log.info("Loading JSON file")
with open(completeFileName, "r") as f:
    data = json.load(f)

for typeName in types:

    logging.info("%s: writing unfiltered files" % typeName)
    unfilteredFileName_train = os.path.join(args.folder, "%s_unfiltered.train.txt" % typeName)
    unfilteredFileName_test = os.path.join(args.folder, "%s_unfiltered.test.txt" % typeName)
    unfilteredFileName_dev = os.path.join(args.folder, "%s_unfiltered.dev.txt" % typeName)

    f_train = open(unfilteredFileName_train, "w")
    f_test = open(unfilteredFileName_test, "w")
    f_dev = open(unfilteredFileName_dev, "w")

    textOnlyCorpus = {}
    textOnlyCorpus['by_document'] = []
    textOnlyCorpus['by_label'] = []

    labelCorpus = []
    textCorpusByLabel = {}

    for record in data:
        if len(record['labels']) == 0:
            continue

        if not typeName in record:
            logging.warning("%s has not %s" % (record['id'], typeName))
            continue

        textOnly = " ".join(record[typeName])
        isTest = record['test'] == 1
        isDev = record['dev'] == 1
        if not isTest and not isDev:
            textOnlyCorpus['by_document'].append(textOnly)

        labelTokens = []
        for label in record['labels']:
            labelNoSpace = label.replace(" ", "_")
            labelTokens.append("__label__" + labelNoSpace)
            if not isTest and not isDev:
                if labelNoSpace not in textCorpusByLabel:
                    textCorpusByLabel[labelNoSpace] = textOnly
                else:
                    textCorpusByLabel[labelNoSpace] += " " + textOnly

        labelText = "\t".join(labelTokens)
        labelCorpus.append(labelText)

        unfilteredText = labelText + "\t" + textOnly + "\n"
        if isTest:
            f_test.write(unfilteredText)
        elif isDev:
            f_dev.write(unfilteredText)
        else:
            f_train.write(unfilteredText)

    f_train.close()
    f_test.close()
    f_dev.close()

    for label in textCorpusByLabel:
        textOnlyCorpus['by_label'].append(textCorpusByLabel[label])

    logging.info("%s: %d documents" % (typeName, len(textOnlyCorpus['by_document'])))
    logging.info("%s: %d labels" % (typeName, len(textOnlyCorpus['by_label'])))

    for corpusName in textOnlyCorpus:
        logging.info("%s-%s: extracting frequencies" % (typeName, corpusName))
        frequencies = {}

        for text in textOnlyCorpus[corpusName]:
            parts = text.split(" ")
            for part in parts:
                if not re.search('[a-zA-Z]', part):
                    continue
                if part not in frequencies:
                    frequencies[part] = 0
                frequencies[part] += 1

        logging.info("%s-%s: extracting stopwords" % (typeName, corpusName))
        stopwords = set()
        for word in frequencies:
            freq = frequencies[word]
            if freq < minFreq:
                stopwords.add(word)

        logging.info("%s-%s: stopwords size: %d" % (typeName, corpusName, len(stopwords)))

        logging.info("%s-%s: removing stopwords" % (typeName, corpusName))
        cleanCorpus = []
        for text in textOnlyCorpus[corpusName]:
            thisList = []
            parts = text.split(" ")
            for part in parts:
                if part not in stopwords:
                    thisList.append(part)
            cleanCorpus.append(" ".join(thisList))

        logging.info("%s-%s: calculating TF-IDF" % (typeName, corpusName))
        vectorizer = TfidfVectorizer(use_idf=True)
        vectorizer.fit_transform(cleanCorpus)

        logging.info("%s-%s: collecting weights" % (typeName, corpusName))
        weigths = {}
        okWords = []
        total = 0.0
        tf_idf = pd.DataFrame(vectorizer.idf_, index=vectorizer.get_feature_names(), columns=["idf_weights"])
        for index, row in tf_idf.iterrows():
            weigths[index] = row['idf_weights']
            if row['idf_weights'] > minWeight[corpusName]:
                okWords.append(index)
            total += row['idf_weights']

        logging.info("%s-%s: ok words size: %d" % (typeName, corpusName, len(okWords)))

        logging.info("%s-%s: writing filtered files" % (typeName, corpusName))
        unfilteredFileName_train = os.path.join(args.folder, "%s_%s_filtered.train.txt" % (typeName, corpusName))
        unfilteredFileName_test = os.path.join(args.folder, "%s_%s_filtered.test.txt" % (typeName, corpusName))
        unfilteredFileName_dev = os.path.join(args.folder, "%s_%s_filtered.dev.txt" % (typeName, corpusName))

        f_train = open(unfilteredFileName_train, "w")
        f_test = open(unfilteredFileName_test, "w")
        f_dev = open(unfilteredFileName_dev, "w")

        for record in tqdm.tqdm(data):
            if len(record['labels']) == 0:
                continue

            if typeName not in record:
                continue

            tokens = record[typeName]
            isTest = record['test'] == 1
            isDev = record['dev'] == 1
            if isTest:
                buffer = f_test
            elif isDev:
                buffer = f_dev
            else:
                buffer = f_train

            if len(record['labels']) == 0:
                continue
            for label in record['labels']:
                labelNoSpace = label.replace(" ", "_")
                buffer.write("__label__")
                buffer.write(labelNoSpace)
                buffer.write("\t")
            inters = intersection(tokens, okWords)
            # if len(inters) == 0:
            #     logging.warning("Zero")
            buffer.write(" ".join(inters))
            buffer.write("\n")

        f_train.close()
        f_test.close()
        f_dev.close()

