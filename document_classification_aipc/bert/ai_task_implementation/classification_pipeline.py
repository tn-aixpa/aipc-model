from kfp import dsl
import mlrun

@dsl.pipeline(
        name='text_classification_bert',
        description='Text Classification with BERT'
    )
def classification_pipeline():
    preprocess = mlrun.run_function(
        'preprocess',
        handler='preprocess_data',
        params={"langs": "it", "data_path": "data/"},
        outputs=[]
    )
    train = mlrun.run_function(
        'train',
        handler='start_train',
        params={"lang": "it", "data_path": "data", "models_path": "models/", "seeds": "all", "epochs": 100, "learning_rate": 3e-5, "threshold": 0.5},
        outputs=[]
    )
    evaluate=mlrun.run_function(
        'evaluate',
        handler="start_evaluate",
        params={"lang": "it", "data_path": "data", "models_path": "models/", "device": "cpu", "batch_size": 8, "threshold": 0.5},
        output=[]
    )

