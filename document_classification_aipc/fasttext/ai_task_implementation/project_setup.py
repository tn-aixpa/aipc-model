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
        func="01-preprocessing_handlers.py",
        handler="parse_ipzs",
        kind="job",
        image="mlrun/mlrun"
    )

    project.set_function(
        name="parsing",
        func="02-parsing_handlers.py",
        handler="parse",
        kind="job",
        image="ertomaselli/classification-parsing:latest"
    )

    project.set_function(
        name="extracting_test",
        func="03-extracting_test_handlers.py",
        handler="extract_test_sets",
        kind="job",
        image="mlrun/mlrun"
    )

    project.set_function(
        name="saving_data",
        func="04-saving_data_handlers.py",
        handler="save_data",
        kind="job",
        image="ertomaselli/classification-parsing:latest"
    )

    project.set_function(
        name="filtering",
        func="05-filtering_handlers.py",
        handler="filter",
        kind="job",
        image="ertomaselli/classification-parsing:latest"
    )

    project.set_function(
        name="training",
        func="training_handlers.py",
        handler="train",
        kind="job",
        image="ertomaselli/classification-training:latest"
    )

    project.set_function(
        name="evaluation",
        func="06-evaluation_handlers.py",
        handler="evaluate",
        kind="job",
        image="mlrun/mlrun"
    )

    project.set_workflow(
        "classification",
        workflow_path="classification_pipeline.py",
        engine="kfp",
        handler="classification_pipeline"
    )

    # Save and return the project:
    project.save()
    return project
