import glob
import json
import os
import zipfile
import tqdm
import logging
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import mlrun
from mlrun import MLClientCtx

# 05-save_fasttext_tfidf.py refactored into callable functions

def filter(context: MLClientCtx, complete_json_file: mlrun.DataItem, minFreq = 5):
    """
    Perform filtering, depending on TF-IDF of words (more common words are filtered out) and generate files for FastText training,  with respect to:
    * Type of tokens: `goodTokens`, `allLemmas`, `allTokens`
    * Filter: `unfiltered` (TF-IDF is not considered), `by_document` (TF-IDF is calculated for each document), `by_label` (TF-IDF is calculated for each label)
    * Role: `train`, `test`, `dev`

    Accepts:
    - complete_json_file: complete JSON file with the information for training
    - minFreq: min frequency for stopwords (default: 5)
    """
    types = ['goodTokens', 'allLemmas', 'allTokens']
    # types = ['goodTokens']

    logging.basicConfig(level=logging.INFO)

    minWeight = {}
    minWeight['by_document'] = 2.0
    minWeight['by_label'] = 1.0

    def intersection(lst1, lst2):
        lst2set = set(lst2)
        lst3 = [value for value in lst1 if value in lst2set]
        return lst3

    #create a folder for the files that will be created, to be zipped and logged as single artifact
    output_folder = "filtering_files"
    os.mkdir(output_folder)

    logging.info("Loading JSON file")
    data = json.loads(complete_json_file.get())

    for typeName in types:

        logging.info("%s: writing unfiltered files" % typeName)

        unfilteredFileName_train = os.path.join(output_folder, "%s_unfiltered.train.txt" % typeName)
        unfilteredFileName_test = os.path.join(output_folder, "%s_unfiltered.test.txt" % typeName)
        unfilteredFileName_dev = os.path.join(output_folder, "%s_unfiltered.dev.txt" % typeName)

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
        # context.log_artifact(unfilteredFileName_train.replace(".txt", ""), local_path=unfilteredFileName_train, upload=True, format="txt")
        # context.log_artifact(unfilteredFileName_test.replace(".txt", ""), local_path=unfilteredFileName_test, upload=True, format="txt")
        # context.log_artifact(unfilteredFileName_dev.replace(".txt", ""), local_path=unfilteredFileName_dev, upload=True, format="txt")

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
            unfilteredFileName_train = os.path.join(output_folder, "%s_%s_filtered.train.txt" % (typeName, corpusName))
            unfilteredFileName_test = os.path.join(output_folder, "%s_%s_filtered.test.txt" % (typeName, corpusName))
            unfilteredFileName_dev = os.path.join(output_folder, "%s_%s_filtered.dev.txt" % (typeName, corpusName))

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
            # context.log_artifact(unfilteredFileName_train.replace(".txt", ""), local_path=unfilteredFileName_train, upload=True, format="txt")
            # context.log_artifact(unfilteredFileName_test.replace(".txt", ""), local_path=unfilteredFileName_test, upload=True, format="txt")
            # context.log_artifact(unfilteredFileName_dev.replace(".txt", ""), local_path=unfilteredFileName_dev, upload=True, format="txt")

    logging.info("Creating zipped folder to log as artifact")
    with zipfile.ZipFile(f"{output_folder}.zip", "w") as f:
        for file in glob.glob(f"{output_folder}/*"):
            f.write(file)

    context.log_artifact(output_folder, local_path=f"{output_folder}.zip", upload=True, format="zip")
