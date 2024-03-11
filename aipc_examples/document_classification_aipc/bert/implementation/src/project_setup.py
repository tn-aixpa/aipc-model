import os
import mlrun
import yaml


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source")
    secrets_file = project.get_param("secrets_file")
    default_image = project.get_param("default_image")
    metadata_path = project.get_param("metadata_path")
    requirements_file = project.get_param("requirements_file")

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
        
    # build a docker image and optionally set it as the project default
 

    # TODO Add the function for preprocessing data for training the model
    
    model_metadata_path = f"{metadata_path}/model.yml"
    with open(model_metadata_path, 'r') as model_md_content:
        models_metadata = yaml.safe_load(model_md_content)
    preproc_fn = project.set_function(
        name="pre-processing",
        func="preprocessing/preprocess.py",
        handler="parse_ipzs",
        kind="job",
    )
    temp = project.code_to_function(
        name="bert-legal2",
        func="functions/train.py",
        handler="start_train",
        kind="job",
        with_repo=True
    )
    project.build_function(temp)
    """
    for model in models_metadata["models"]:
        print(model["training"]["implementation"]["source"])
        print(model["name"])
        print(model["training"]["implementation"]["handler"])

        if model["training"]["implementation"]["source"] != "":
            project.set_function(
                name="bert-legal-acts-classification",
                func="functions/train.py",
                handler="start_train",
                kind="job"
            )
    """

                     
    # TODO define project artifacts such as: upload the training dataset folder
    
    
    # Save and return the project:
    project.save()
    return project
