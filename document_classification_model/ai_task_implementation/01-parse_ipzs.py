import os
import json
import logging
import tqdm
import csv
import re
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='Save file for FastText.')
parser.add_argument("input_folder", metavar="input-folder", help="Folder containing atti_SG_materie.csv and JSON files")
parser.add_argument("output_file", metavar="output-file", help="Output file")

args = parser.parse_args()

idPrefix = "ipzs-"
limit = 10

logging.basicConfig(level=logging.INFO)
csvfilename = os.path.join(args.input_folder, "atti_SG_materie.csv")

codePatt = re.compile("([^_]+_[^_]+)_([^_]+)_([^_]+)\.json")

dayIndex = {}
index = {}

for root, subFolder, files in os.walk(args.input_folder):
    for item in files:
        if item.endswith(".json") :
            fileNamePath = str(os.path.join(root,item))
            m = codePatt.match(item)
            if m:
                if m.group(1) in index:
                    logging.info(m.group(1) + " already exists")
                    logging.info(fileNamePath + " --- " + index[m.group(2)])
                index[m.group(1)] = fileNamePath
            else:
                logging.error("ERR: " + item)

allObjs = []
count = {}

with open(csvfilename, "r") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(csvreader, None)
    for row in csvreader:
        day = row[1]
        code = row[0]
        category = row[3]
        d = datetime.strptime(day.lower(), "%d-%b-%y").date()
        code = datetime.strftime(d, "%Y%m%d") + "_" + code
        if not code in index:
            logging.error("%s not in index" % code)
            continue

        with open(index[code], "r") as f:
            data = json.load(f)
            titolo = data['metadati']['titoloDoc']
            titolo = re.sub(r"\n", " ", titolo)
            titolo = re.sub(r" +", " ", titolo)
            thisObj = {}
            thisObj['text'] = titolo
            thisObj['id'] = idPrefix + code
            thisObj['labels'] = [category]
            if category not in count:
                count[category] = 0
            count[category] += 1
            allObjs.append(thisObj)

print(len(allObjs))
if limit > 1:
    for index in count:
        if count[index] < limit:
            allObjs = [x for x in allObjs if index not in x['labels']]

print(len(allObjs))
jsonString = json.dumps(allObjs, indent=4)
jsonFile = open(args.output_file, "w")
jsonFile.write(jsonString)
jsonFile.close()

