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


    #serving_fn = project.set_function(
    #    model["inference"]["serving"]["implementation"]["source"],
    #    name="serving-predictor",
    #    kind="serving",
    #    requirements=model["inference"]["serving"]["implementation"][
    #        "requirements"
    #    ],
    #)

    # TODO define project artifacts such as: upload the training dataset folder

    # Save and return the project:
    project.save()
    return project
