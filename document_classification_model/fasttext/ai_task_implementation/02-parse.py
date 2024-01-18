import argparse

parser = argparse.ArgumentParser(description='Parse files with Tint.')
parser.add_argument("input_file", metavar="input-file", help="JSON complete file with labels and IDs")
parser.add_argument("output_folder", metavar="output-folder", help="Output folder")
parser.add_argument("--tint_url", help="Tint URL", type=str)

args = parser.parse_args()

import requests
import json
import os
import tqdm
import logging
import pprint
import utils

logging.basicConfig(level=logging.INFO)

if not args.tint_url:
    import stanza
    nlp = stanza.Pipeline('it', processors='tokenize,mwt,pos,lemma') # en , oppure altre lingue : spaCy

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

logging.info("Loading texts")
with open(args.input_file, "r") as f:
    data = json.load(f)

errors = []

logging.info("Performing NLP")
pbar = tqdm.tqdm(data, smoothing=0.5, maxinterval=1)
for record in pbar:
    if isinstance(record, str):
        record = data[record]
    text = record['text']
    thisId = record['id']
    pbar.set_description(thisId)
    outputFile = os.path.join(args.output_folder, thisId + ".json")
    if os.path.exists(outputFile):
        continue
    try:
        if args.tint_url:
            parsedData = utils.runTint(args.tint_url, text)
        else:
            parsedData = utils.runStanza(nlp, text)
    except Exception:
        print(thisId)
        errors.append(thisId)
        continue
    with open(outputFile, 'w') as fw:
        fw.write(json.dumps(parsedData, indent = 4))

logging.info("Errors: " + str(errors))
