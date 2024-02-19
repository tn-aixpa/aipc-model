import os
import requests
import json
import tqdm
import logging
import zipfile
import glob
import mlrun
from mlrun.execution import MLClientCtx

# 02-parse.py refactored into callable functions
# utils.py functions are here instead of in a separate module

def parse(context: MLClientCtx, input_file: mlrun.DataItem, tint_url = None):
    """
    Parse files with Tint or Stanza to lemmatize the documents and extract part-of-speeches.
    
    A JSON file is generated for each record in input_file. Accepts:
    - input_file: JSON file with labels and IDs
    - tint_url: Tint URL (if not provided, Stanza is used by default)
    """
    logging.basicConfig(level=logging.INFO)

    if not tint_url:
        import stanza
        nlp = stanza.Pipeline('it', processors='tokenize,mwt,pos,lemma')

    #create a folder for the JSON files that will be created, to be zipped and logged as single artifact
    output_folder = "tint_files"
    os.mkdir(output_folder)

    logging.info("Loading texts")
    data = json.loads(input_file.get())

    errors = []

    logging.info("Performing NLP")
    pbar = tqdm.tqdm(data, smoothing=0.5, maxinterval=1)

    #logged_artifacts = []

    for record in pbar:
        if isinstance(record, str):
            record = data[record]
        text = record['text']
        thisId = record['id']
        pbar.set_description(thisId)

        # if thisId in logged_artifacts:
        #     continue #avoids duplicates in input file (i.e. current run) but does not preserve results from previous runs

        outputFile = os.path.join(output_folder, thisId + ".json")
        if os.path.exists(outputFile):
            continue
        try:
            if tint_url:
                parsedData = _runTint(tint_url, text)
            else:
                parsedData = _runStanza(nlp, text)
        except Exception:
            print(thisId)
            errors.append(thisId)
            continue
        with open(outputFile, 'w') as fw:
            fw.write(json.dumps(parsedData, indent = 4))
        # jsonString = json.dumps(parsedData, indent = 4)
        # artifact = context.log_artifact(thisId, body=jsonString, local_path=outputFile, upload=True, format="json")
        # logged_artifacts.append(artifact.metadata.key)

    logging.info("Creating zipped folder to log as artifact")
    with zipfile.ZipFile(f"{output_folder}.zip", "w") as f:
        for file in glob.glob(f"{output_folder}/*"):
            f.write(file)

    context.log_artifact(output_folder, local_path=f"{output_folder}.zip", upload=True, format="zip")

    logging.info("Errors: " + str(errors))

def _runStanza(nlp, sentence_text):
    doc = nlp(sentence_text)
    data = {}
    data['sentences'] = []
    sentID = 0
    for sentence in doc.sentences:
        newSent = {}
        newSent['tokens'] = []
        newSent['index'] = sentID
        sentID += 1
        for token in sentence.tokens:
            if len(token.words) == 1:
                newToken = {}
                newToken['index'] = token.id[0]
                newToken['originalText'] = token.text
                newToken['word'] = token.text
                newToken['featuresText'] = "_"
                if hasattr(token, "feats"):
                    newToken['featuresText'] = token.feats
                newToken['characterOffsetBegin'] = token.start_char
                newToken['characterOffsetEnd'] = token.end_char
                newToken['isMultiwordToken'] = False
                newToken['isMultiwordFirstToken'] = False
                for t in token.words:
                    newToken['pos'] = t.xpos
                    newToken['ud_pos'] = t.upos
                    newToken['lemma'] = t.lemma
                newSent['tokens'].append(newToken)
            else:
                first = True
                for t in token.words:
                    newToken = {}
                    newToken['index'] = t.id
                    newToken['originalText'] = token.text
                    newToken['word'] = t.text
                    newToken['characterOffsetBegin'] = token.start_char
                    newToken['characterOffsetEnd'] = token.end_char
                    newToken['featuresText'] = "_"
                    if hasattr(t, "feats"):
                        newToken['featuresText'] = t.feats
                    newToken['isMultiwordToken'] = True
                    newToken['isMultiwordFirstToken'] = first
                    newToken['pos'] = t.xpos
                    newToken['ud_pos'] = t.upos
                    newToken['lemma'] = t.lemma
                    first = False
                    newSent['tokens'].append(newToken)

        data['sentences'].append(newSent)
    return data

def _runTint(tint_url, sentence_text):
    # requires args['tint-url']
    myobj = {'text' : sentence_text.strip()}
    x = requests.post(tint_url, data = myobj)
    data = json.loads(x.text)
    return data
