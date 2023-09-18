# Document classification scripts

This set of Python script can be used to train a multi-class multi-label classifier for documents.
There are two implementation: FastText and BERT (work in progress).

## Installation

Just run `pip install -r requirements.txt` to install all the libraries needed to run the scripts.

## 1. Pre-processing

Depending on the dataset, there are different Python scripts that can be used to preprocess the data. Once run the right one, data is saved in a unique format, so that the rest of the scripts do not need to be customized.

The scripts which names start with `01` refer to the pre-processing phase where data in any format is converted to a JSON file structured as follows:

```
{
    "20000111_000G0003": {
        "text": "Interventi straordinari nel settore dei beni e delle attivita' culturali. ",
        "id": "ipzs-ml-20000111_000G0003",
        "labels": [
            "A0400",
            "C0I90"
        ]
    },
...
}
```

Each record must have an ID, that is the key of the JSON record, a text and a list of labels.

The JSON output file should be saved into a dedicated folder, just to facilitate the following steps. For example:
```
mkdir output-folder
python3 01-parse_ipzs.py input-folder output-folder/data.json
```

## 2. Parsing

The script called `02-parse.py` run Tint or Stanza to lemmatize the documents, extract part-of-speeches, and save a JSON files for each document.
By default, Stanza is used, unless the `--tint_url` parameter is used.

Example:
```
python3 02-parse.py output-folder/data.json output-folder/parsed
```

## 3. Selecting test data

Two balanced (with respect to labels) set of documents are selected and the list of corresponding IDs are saved into two files (dev and test).
The files are named `testlist.txt` and `devlist.txt`.
Parameters `--test-ratio` and `--dev-ratio` can be set (between 0 and 1) to select the amount of documents for each set.

Example:
```
python3 03-extract_test.py output-folder/parsed output-folder/data.json output-folder/
```

## 4. Save data

A file named `complete.json` is saved including all the information needed for training: ID, text, lemmas, content words, labels, test/dev.

Example:
```
python3 04-save_data.py output-folder/parsed output-folder/data.json output-folder/
```

## 5. Filtering

Some filtering is performed, depending on TF-IDF of words (more common words are filtered out).
Files are saved (ready for FastText training) with respect to:
* Type of tokens: `goodTokens`, `allLemmas`, `allTokens`.
* Filter: `unfiltered`, `by_document`, `by_label` (respectively depending on whether the TF-IDF is not considered, is calculated for each document or for each label).
* Role: `train`, `test`, `dev`.

Example:
```
python3 05-save_fasttext_tfidf.py output-folder/
```

## Training

Training is performed using FastText. Some helper shell scripts are already provided. In particular, `run-fasttext.sh` runs everything (train and test).
See inside the `.sh` scripts to check what is executed.

## 6. Calculating micro and macro F1

The final script `06-micro_macro.py` can be used to get information on the performances of the trained model.
The two mandatory parameters are `pred_file` and `gold_file` (in FastText format).
One can also set `--prob-threshold` (the threshold below which the prediction is discarded) and `--show-cm` to show the confusion matrix.

## Run the server

Once the models are created and the best one is chosen, the `server.py` can be used to run the HTTP server that will perform the classification for new texts.

```
python3 server.py [port] [model_file] [label_conversion]
```

where `label_conversion` is a file to convert labels into textual meaning (CSV file, the third column being the ID and the fourth one being the description).
Default parser is Stanza. Tint can be used by setting `--tint_host` and `--tint_port` arguments.

The server needs to be called using a POST request, text being the body.
