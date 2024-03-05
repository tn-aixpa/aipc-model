import os
import mlrun

def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source")
    secrets_file = project.get_param("secrets_file")
    default_image = project.get_param("default_image")

    # Set project git/archive source and enable pulling latest code at runtime
    if source:
        print(f"Project Source: {source}")
        project.set_source(project.get_param("source"), pull_at_runtime=True)

    # Create project secrets and also load secrets in local environment
    if secrets_file and os.path.exists(secrets_file):
        project.set_secrets(file_path=secrets_file)
        mlrun.set_env_from_file(secrets_file)

    # Set default project docker image - functions that do not specify image will use this
    if default_image:
        project.set_default_image(default_image)

    project.set_function(
        name="pre-processing",
        func="./preprocessing/01-preprocessing.py",
        handler="parse_ipzs",
        kind="job",
    )

    project.set_function(
        name="parsing",
        func="./preprocessing/02-parsing.py",
        handler="parse",
        kind="job",
    )

    project.set_function(
        name="extracting-test",
        func="./preprocessing/03-extracting_test.py",
        handler="extract_test_sets",
        kind="job",
    )

    project.set_function(
        name="saving-data",
        func="./preprocessing/04-saving_data.py",
        handler="save_data",
        kind="job",
    )

    project.set_function(
        name="filtering",
        func="./preprocessing/05-filtering.py",
        handler="filter",
        kind="job",
    )
    
    project.set_function(
        name="training",
        func="./functions/model_training.py",
        handler="train",
        kind="job"
    )

    project.set_function(
        name="evaluation",
        func="./functions/model_evaluation.py",
        handler="evaluate",
        kind="job"
    )

    project.set_workflow(
        "classification",
        workflow_path="./workflows/main_workflow.py",
        engine="kfp",
        handler="classification_pipeline"
    )
    
    # Save and return the project:
    project.save()
    return project
