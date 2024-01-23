import json
import logging
import fasttext
import stanza
import pandas as pd
import mlrun

# NOTE: code partially taken by server.py, which seems outdated (e.g. accesses variables that do not exist) and not tested with Stanza.
# Predict is called directly on the POSTed text, as it is not clear why the original code processed it with Stanza before predicting.

class ClassifierModel(mlrun.serving.V2ModelServer):
    def load(self):
        """Download and initialize the model and/or other elements"""
        logging.basicConfig(level=logging.INFO)
        model_file, extra_data = self.get_model(".bin")
        self.model = fasttext.load_model(model_file)

        logging.info(f"extra_data: {extra_data}")

    def predict(self, body: dict) -> list:
        """Accept request payload and return prediction/inference results (called when you access /infer or /predict)
        Body has the following structure:
        {
            "id" : $string #optional, unique Id of the request, if not provided a random value is provided
            "model" : $string #optional, model to select (for streaming protocols without URLs)
            "data_url" : $string #optional, option to load the inputs from an external file/s3/v3io/… object
            "parameters" : $parameters #optional, optional request parameters
            "inputs" : [ $request_input, ... ], #list of input elements (numeric values, arrays, or dicts)
            "outputs" : [ $request_output, ... ] #optional, requested output values
        }
        """
        logging.basicConfig(level=logging.INFO)
        limit = 0.01

        nlp = stanza.Pipeline('it', processors='tokenize,mwt,pos,lemma')

        print('received files')
        print(body)
        csvfilename = body["inputs"][0] #File atti_SG_materie.csv
        text = body["inputs"][1]

        csvfile = pd.read_csv(csvfilename, quotechar='"').to_dict(orient="records")

        materie_dict = {}
        for row_dict in csvfile:
            materie_dict[row_dict["CODMAT"]] = row_dict["TMAT"]

        logging.info(f"materie_dict: {materie_dict}")

        res = self.model.predict(text, k=-1)
        logging.info(f"res: {res}")

        data = _runStanza(nlp, text)
        logging.info(f"data: {data}")

        out = {}
        out['topics'] = {}
        out['words'] = data['sentences']
        for i in range(len(res[0])):
            if res[1][i] > limit:
                title = res[0][i].replace('__label__', "")
                title += " - " + materie_dict[title]
                out['topics'][title] = res[1][i]

        return [json.dumps(out)]


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
