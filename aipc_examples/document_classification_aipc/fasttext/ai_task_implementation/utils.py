import yaml
import requests
import zipfile
import io
import pandas as pd

def load_validation_card_specifications():
    """
    Load the specifications for the validation card.

    Returns:
    dict: The loaded validation card specifications.
    """
    with open("cards_metadata/validation_card.yml", "r") as valid_card_data:
        valid_card = yaml.safe_load(valid_card_data)        
        #reference_data = valid_card["reference_data"]
        #legal_acts_data = read_dataset(reference_data)
        return valid_card


def load_data_card_specifications():
    """
    Load the specifications for the data card.

    Returns:
    dict: The loaded validation card specifications.
    """
    with open("cards_metadata/data_card.yml", "r") as data_card_data:
        valid_card = yaml.safe_load(data_card_data)  
        return valid_card


def read_reference_dataset(reference_data):
    legal_acts_content = requests.get(reference_data).content
    with zipfile.ZipFile(io.BytesIO(legal_acts_content)) as arc:
        legal_acts_data = pd.read_json(arc.open("data.json"))
        legal_acts_data['labels'] = legal_acts_data['labels'].apply(lambda x: ','.join(map(str, x)))
        legal_acts_data['year'] = legal_acts_data['id'].str.slice(5, 9).astype(int)
    return legal_acts_data

def read_current_dataset(data_path):
    return pd.read_json(data_path)