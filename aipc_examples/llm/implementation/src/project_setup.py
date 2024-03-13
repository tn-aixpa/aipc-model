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
    if requirements_file:
        project.build_image(
            base_image=default_image,
            requirements_file=requirements_file,
            set_as_default=True
        )

    model_metadata_path = f"{metadata_path}/model.yml"
    with open(model_metadata_path, 'r') as model_md_content:
        models_metadata = yaml.safe_load(model_md_content)
    
    for model in models_metadata["models"]:
        if model["training"]["implementation"]["source"]:
            project.set_function(
                name=model["name"],
                func=model["training"]["implementation"]["source"],
                handler=model["training"]["implementation"]["handler"],
                kind="job"
            )
        if model["evaluation"]["implementation"]["source"]:
            project.set_function(
                name=model["name"],
                func=model["training"]["implementation"]["source"],
                handler=model["training"]["implementation"]["handler"],
                kind="job"
            )
    if model["inference"]["serving"]["implementation"]["source"]:
        serving_fn = project.set_function(
            model["inference"]["serving"]["implementation"]["source"], 
            name='serving-predictor', 
            kind='serving',
            requirements=model["inference"]["serving"]["implementation"]["requirements"]
        )    
        serving_fn.add_model(
            "serving_model",
            model_path=model["training"]["output_dir"],
            class_name=model["inference"]["serving"]["implementation"]["class_name"],
        )    
        project.deploy_function(serving_fn)
        
    # TODO define project artifacts such as: upload the training dataset folder
    
    
    # Save and return the project:
    project.save()
    return project
