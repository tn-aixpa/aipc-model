from kfp import dsl
import mlrun
import yaml
from drift_detection_evidently import build_drift_detection_test
from utils import load_validation_card_specifications

@dsl.pipeline(
    name="classification-pipeline",
    description="Document classification pipeline"
)
def classification_pipeline(
        # data_format: str,
        # bucket_name: str, idPrefix = "ipzs-", limit = 10, max_documents = None,
        # tint_url = None,
        # testRatio = 0.2, devRatio = 0.2,
    ):
    """
    Accepts:
    - project_name_or_url: name (in DB) or git or tar.gz or .zip sources archive path
    - data_format: parse_ipzs, parse_ipzs_ml, parse_expmat, parse_esco, parse_wemapp, parse_wikinews

    The additional parameters are documented in the handlers definitions.
    """
    project = mlrun.get_current_project()

    #run preprocessing function, setting the chosen handler
    preprocess = project.run_function(
        "pre-processing",
        #handler=data_format, #choose handler depending on dataset
        params={"bucket_name": "ipzs", "idPrefix": "ipzs-", "limit": 10, "max_documents": 250},
        outputs=["preprocessed_data"]
    )

    #run parsing function with default handler
    parse = project.run_function(
        "parsing",
        inputs={"input_file": preprocess.outputs["preprocessed_data"]},
        params={"tint_url": None},
        outputs=["tint_files"]
    )

    #run test extraction function with default handler
    extract_test = project.run_function(
        "extracting_test",
        inputs={"input_file": preprocess.outputs["preprocessed_data"], "tint_files": parse.outputs["tint_files"]},
        params={"testRatio": 0.2, "devRatio": 0.2},
        outputs=["testlist", "devlist"]
    )

    #run data saving function with default handler
    save = project.run_function(
        "saving_data",
        inputs={"input_file": preprocess.outputs["preprocessed_data"],
                "tint_files": parse.outputs["tint_files"],
                "test_list_file": extract_test.outputs["testlist"],
                "dev_list_file": extract_test.outputs["devlist"]},
        outputs=["complete"]
    )

    #run filtering function with default handler
    filter = project.run_function(
        "filtering",
        inputs={"complete_json_file": save.outputs["complete"]},
        params={"minFreq": 3},
        outputs=["filtering_files"]
    )

    # load configuration for data drifting/quality checks
    reference_data = load_validation_card_specifications()
    current_data = filter.outputs["filtering_files"]  

    # add conditional run of training based on the results of the drifting test suites
    if build_drift_detection_test(reference_data, current_data):
    #run training function with default handler
        train = project.run_function(
            "training",
            inputs={"training_files": filter.outputs["filtering_files"]},
            outputs=["results"]
        )

        #run evaluation function with default handler
        evaluate = project.run_function(
            "evaluation",
            inputs={"pred_files": train.outputs["results"], "gold_files": filter.outputs["filtering_files"]},
            params={"show_cm": True}
        )




# def pipeline(label_column: str, test_size=0.2):
    
#     # Ingest the data set
#     ingest = mlrun.run_function(
#         'get-data',
#         handler='prep_data',
#         params={'label_column': label_column},
#         outputs=["iris_dataset"]
#     )
    
#     # Train a model   
#     train = mlrun.run_function(
#         "train-model",
#         handler="train_model",
#         inputs={"dataset": ingest.outputs["iris_dataset"]},
#         params={
#             "label_column": label_column,
#             "test_size" : test_size
#         },
#         outputs=['model']
#     )
    
#     # Deploy the model as a serverless function
#     deploy = mlrun.deploy_function(
#         "deploy-model",
#         models=[{"key": "model", "model_path": train.outputs["model"]}]
#     )
