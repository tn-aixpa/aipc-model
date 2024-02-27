import logging
import json
import math
import random
import zipfile
import os
import mlrun
from mlrun.execution import MLClientCtx

# 03-extract_test.py refactored into callable functions

def extract_test_sets(context: MLClientCtx, input_file: mlrun.DataItem, tint_files: mlrun.DataItem, testRatio = 0.2, devRatio = 0.2):
    """
    Select two balanced (with respect to labels) sets of documents and save the corresponding IDs in `testlist.txt` and `devlist.txt`. Accepts:
    - input_file: JSON file with labels and IDs
    - tint_files: collection of JSON files parsed by Tint or Stanza
    - testRatio: percentage of test data (default: 0.2)
    - devRatio: percentage of dev data (default: 0.2)
    """
    logging.basicConfig(level=logging.INFO)

    testListName = "testlist.txt"
    devListName = "devlist.txt"

    if testRatio + devRatio >= 1:
        logging.error("dev + test must be less than 1")
        return

    if testRatio + devRatio >= 0.8:
        logging.warning("dev + test seems too big")

    logging.info("Loading JSON file")
    data = json.loads(input_file.get())

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

    logging.info("Downloading and extracting tint_files.zip")
    tint_files.download(target_path="tint_files.zip")

    with zipfile.ZipFile("tint_files.zip", "r") as z:
        z.extractall() #extracts a folder named tint_files

    for record in data:
        # Better to check this here, too
        # if not record['id'] in tint_files:
        #     logging.warning("File %s does not exist" % record['id'])
        #     continue
        tintFile = os.path.join("tint_files", record['id'] + ".json")
        if not os.path.exists(tintFile):
            logging.warning("File %s does not exist" % tintFile)
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
                logging.warning("Unbalanced training ratio for class %s (total: %d): %.2f - test: %d - dev: %d - train: %d" %
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

        logging.info("Shuffling data")
        random.shuffle(data)

        logging.info("Extracting texts")
        for record in data:
            tintFile = os.path.join("tint_files", record['id'] + ".json")
            if not os.path.exists(tintFile):
                logging.warning("File %s does not exist" % tintFile)
                continue

            if len(record['labels']) == 0:
                logging.warning("File %s has no labels, skipping" % tintFile)
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

    logging.info("Saving test list")
    with open(testListName, "w") as fw:
        for docID in testList:
            fw.write(docID)
            fw.write("\n")
    context.log_artifact("testlist", local_path=testListName, upload=True, format="txt")

    logging.info("Saving dev list")
    with open(devListName, "w") as fw:
        for docID in devList:
            fw.write(docID)
            fw.write("\n")
    context.log_artifact("devlist", local_path=devListName, upload=True, format="txt")

    logging.info("Train size: %d" % trainSize)
    logging.info("Test size: %d" % testSize)
    logging.info("Dev size: %d" % devSize)
