import mlrun

project = mlrun.get_or_create_project("text_classification_fasttext", context='./')
project.set_workflow(
    "classification",
    workflow_path="../ai_task_implementation/classification_pipeline.py",
    engine="kfp",
    handler="classification_pipeline"
)
project.save()
run_id = project.run(
    name="classification"
)






