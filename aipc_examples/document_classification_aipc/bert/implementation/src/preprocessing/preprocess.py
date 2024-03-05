import argparse
import yaml
import os
import mlrun
from utils8 import process_datasets
from mlrun import MLClientCtx

seeds = []

@mlrun.handler()
def preprocessdata(
        context: MLClientCtx, 
        data_path: str = "data/", 
        langs="it", 
        years: str = "all",
        seeds_value: str = "110",
        max_length: int = 512,
        limit_tokenizer: bool = False,
        get_doc_ids:bool = False,
        add_mt_do:bool = False,
        title_only:bool = False,
        add_title:bool = False
        ):
    """
    Load the configuration file and process the data.
    :param langs: Languages to be processed, separated by a comme (e.g. en,it). Write 'all' to process all the languages.
    :param data_path: Path to the data to process.
    :param years: Year range to be processed, separated by a minus (e.g. 2010-2020 will get all the years between 2010 and 2020 included) or individual years separated by a comma (to use a single year, simply type it normally like '2016'). Write 'all' to process all the files in the folder.
    :param seeds: Seeds to be used for the randomization and creating the data splits, separated by a comma (e.g. 110,221).
    :param add_title: Add the title to the text.
    :param title_only: Use only the title as input.
    :param max_length: Maximum number of words of the text to be processed.
    :param limit_tokenizer: Limit the tokenizer length to the maximum number of words. This will remove the statistics for the documents length.
    :param add_mt_do: Add the MicroThesaurus and Domain labels to be predicted.
    :param get_doc_ids: Get the document ids that are used in the splits. NOTE: only use for debugging.
    """
    with open("config/models.yml", "r") as fp:
        config = yaml.safe_load(fp)
    
    seeds = seeds_value.split(",")

    print(f"Tokenizers config:\n{format(config)}")
    
    for directory in os.listdir(data_path):
        # If we specified one or more languages, we only process those.
        if langs != "all" and directory not in langs.split(","):
            continue
        
        print(f"\nWorking on directory: {format(directory)}...")
        lang = directory
        print(f"Lang: '{lang}', Tokenizer: '{config[lang]}'")

        process_datasets(data_path, directory, config[lang], years, max_length, limit_tokenizer, get_doc_ids, add_mt_do, title_only, add_title, seeds)

    #context.log_artifact(output_folder, local_path=f"{output_folder}.zip", upload=True, format="zip")

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(preprocessdata)