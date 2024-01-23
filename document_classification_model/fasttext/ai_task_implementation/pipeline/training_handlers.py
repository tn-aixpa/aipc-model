import glob
import os
import zipfile
import fasttext
import pandas as pd
import mlrun
from mlrun import MLClientCtx
import logging

# run-fasttext.sh refactored into callable functions, whithout calls to 06-micro_macro.py

def train(context: MLClientCtx, training_files: mlrun.DataItem):
    logging.basicConfig(level=logging.INFO)

    types = ["goodTokens", "allLemmas", "allTokens"]
    bys = ["by_document", "by_label"]

    logging.info("Downloading and extracting filtering_files.zip")
    training_files.download(target_path="filtering_files.zip")

    with zipfile.ZipFile("filtering_files.zip", "r") as z:
        z.extractall() #extracts a folder named filtering_files

    training_files_folder = "filtering_files"

    if len(os.listdir(training_files_folder)) == 0:
        logging.info("No data supplied")
        return

    #create a folder for the result files that will be created, to be zipped and logged as single artifact
    results_folder = "results"
    os.mkdir(results_folder)

    for type in types:
        unfiltered_train = os.path.join(training_files_folder, f"{type}_unfiltered.train.txt")
        unfiltered_test = os.path.join(training_files_folder, f"{type}_unfiltered.test.txt")
        unfiltered_model = f"{type}_unfiltered_model.bin"
        unfiltered_results = os.path.join(results_folder, f"{type}_unfiltered.results.txt")

        # mlrun.get_dataitem(training_files[f"{type}_unfiltered.train"]).download(target_path=unfiltered_train)
        # mlrun.get_dataitem(training_files[f"{type}_unfiltered.test"]).download(target_path=unfiltered_test)

        logging.info(f"Creating {type} model")
        model = fasttext.train_supervised(input=unfiltered_train, epoch=25, lr=1.0)
        model.save_model(unfiltered_model)

        logging.info(f"Testing {type} model")
        model.test(unfiltered_test)

        logging.info(f"Predicting with {unfiltered_model} model for the whole test data")
        test_df = pd.read_csv(unfiltered_test, sep="\t", header=None, names=["labels", "texts"])
        with open(unfiltered_results, "w") as f:
            for line in test_df["texts"]:
                predict_tuple = model.predict(line)
                label = predict_tuple[0][0]
                prob = predict_tuple[1][0]
                f.write(label + "\t" + str(prob) + "\n")

        logging.info(f"Logging {type} artifacts")
        context.log_model(unfiltered_model.replace(".bin", ""), model_file=unfiltered_model, upload=True, framework="fastText")
        #context.log_artifact(unfiltered_results.replace(".txt", ""), local_path=unfiltered_results, upload=True, format="txt")

        for b in bys:
            filtered_train = os.path.join(training_files_folder, f"{type}_{b}_filtered.train.txt")
            filtered_test = os.path.join(training_files_folder, f"{type}_{b}_filtered.test.txt")
            filtered_model = f"{type}_{b}_filtered_model.bin"
            filtered_results = os.path.join(results_folder, f"{type}_{b}_filtered.results.txt")

            # mlrun.get_dataitem(training_files[f"{type}_{b}_filtered.train"]).download(target_path=filtered_train)
            # mlrun.get_dataitem(training_files[f"{type}_{b}_filtered.test"]).download(target_path=filtered_test)

            logging.info(f"Creating {type}_{b} model")
            model = fasttext.train_supervised(input=filtered_train, epoch=25, lr=1.0)
            model.save_model(filtered_model)

            logging.info(f"Testing {type}_{b} model")
            model.test(filtered_test)

            logging.info(f"Predicting with {filtered_model} model for the whole test data")
            test_df = pd.read_csv(filtered_test, sep="\t", header=None, names=["labels", "texts"])
            with open(filtered_results, "w") as f:
                for line in test_df.fillna("")["texts"]:
                    #filling NaN with empty string, happens when no okWords are included during filtering
                    #NOTE: this means some prediction may be performed on an empty string
                    predict_tuple = model.predict(line)
                    label = predict_tuple[0][0]
                    prob = predict_tuple[1][0]
                    f.write(label + "\t" + str(prob) + "\n")

            logging.info(f"Logging {type}_{b} artifacts")
            context.log_model(filtered_model.replace(".bin", ""), model_file=filtered_model, upload=True, framework="fastText")
            #context.log_artifact(filtered_results.replace(".txt", ""), local_path=filtered_results, upload=True, format="txt")

    logging.info("Creating zipped folder to log as artifact")
    with zipfile.ZipFile(f"{results_folder}.zip", "w") as f:
        for file in glob.glob(f"{results_folder}/*"):
            f.write(file)

    context.log_artifact(results_folder, local_path=f"{results_folder}.zip", upload=True, format="zip")
