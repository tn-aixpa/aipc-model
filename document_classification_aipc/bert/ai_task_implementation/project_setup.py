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
        "preprocess.py",
        name="preprocess",
        kind="job",
        image="mlrun/mlrun",
        handler="preprocessdata"
    )

    project.set_function(
        "train.py",
        name="train",
        kind="job",
        image="mlrun/mlrun",
        handler="train"
    )

    project.set_function(
        "evaluate.py",
        name="evaluate",
        kind="job",
        handler="evaluate"
    )

    # Save and return the project:
    project.save()
    return project