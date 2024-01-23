import yaml

def load_validation_card_specifications():
    """
    Load the specifications for the validation card.

    Returns:
    dict: The loaded validation card specifications.
    """
    with open("cards_metadata/validation_card.yml", "r") as valid_card_data:
        valid_card = yaml.safe_load(valid_card_data)
        reference_data = valid_card["reference_data"]
        return reference_data