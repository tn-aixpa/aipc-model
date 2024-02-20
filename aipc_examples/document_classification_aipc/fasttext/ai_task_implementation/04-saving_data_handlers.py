import json
import os
import zipfile
import tqdm
import logging
import mlrun
from mlrun.execution import MLClientCtx

# 04-save_data.py refactored into callable functions

def save_data(context: MLClientCtx, input_file: mlrun.DataItem, tint_files: mlrun.DataItem, test_list_file: mlrun.DataItem, dev_list_file: mlrun.DataItem):
    """
    Store in a JSON file all the information needed for training: ID, text, lemmas, content words, labels, test/dev. Accepts:
    - input_file: JSON file with labels and IDs
    - tint_files: collection of JSON files parsed by Tint or Stanza
    - test_list_file: list of test document IDs
    - dev_list_file: list of dev document IDs
    """
    logging.basicConfig(level=logging.INFO)

    completeFileName = "complete.json"

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

    logging.info("Loading JSON file")
    data = json.loads(input_file.get())

    if isinstance(data, dict):
        data_tmp = []
        for record in data:
            data_tmp.append(data[record])
        data = data_tmp

    testList = test_list_file.get(encoding="utf-8").split()
    devList = dev_list_file.get(encoding="utf-8").split()

    logging.info("Downloading and extracting tint_files.zip")
    tint_files.download(target_path="tint_files.zip")

    with zipfile.ZipFile("tint_files.zip", "r") as z:
        z.extractall() #extracts a folder named tint_files

    logging.info("Extracting texts")
    textOnlyCorpus = []
    for record in tqdm.tqdm(data):
        tintFile = os.path.join("tint_files", record['id'] + ".json")
        if not os.path.exists(tintFile):
            logging.warning("File %s does not exist" % tintFile)
            continue

        if len(record['labels']) == 0:
            logging.warning("File %s has no labels, skipping" % tintFile)
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

    logging.info("Saving file")
    jsonString = json.dumps(data, indent = 4)
    context.log_artifact("complete", body=jsonString, local_path=completeFileName, upload=True, format="json")
