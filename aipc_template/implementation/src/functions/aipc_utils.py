import yaml

def load_model_card_specifications():
    """
    Load the specifications for the model card.

    Returns:
    dict: The loaded model card specifications.
    """
    with open("../../metadata/model.yml", "r") as model_card_data:
        model_card = yaml.safe_load(model_card_data)        
        return model_card
    
    
def load_validation_card_specifications():
    """
    Load the specifications for the card.

    Returns:
    dict: The loaded card specifications.
    """
    with open("../../metadata/validation.yml", "r") as card_data:
        card = yaml.safe_load(card_data)        
        return card