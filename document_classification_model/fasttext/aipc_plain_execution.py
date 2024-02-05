import sys
import os
import importlib.util
from ai_task_implementation.pipeline import utils

def run_drift_detection():
    validation_card_metadata = utils.load_validation_card_specifications()

    specification = validation_card_metadata["specification"]
    implementation_file = specification["implementation"]
    package = specification["package"] 
    handler = specification["handler"]

    reference_data = utils.read_reference_dataset(validation_card_metadata["reference_data"])
    current_data = utils.read_current_dataset(validation_card_metadata["input_data"])

    implementation_module = importlib.import_module(package + "." + implementation_file)
    implementation = getattr(implementation_module, handler)
    result = implementation(reference_data, current_data)
    
    print(result)



if __name__=="__main__":
    action = sys.argv[1]
    actions = {
        "run_drift_detection": run_drift_detection
    }
    actions[action]()

    
