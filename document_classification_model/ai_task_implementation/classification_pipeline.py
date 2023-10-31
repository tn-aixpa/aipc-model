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
        params={},
        outputs=[]
    )
    train = mlrun.run_function(
        'train',
        handler='start_train',
        params={},
        outputs=[]
    )
    evaluate=mlrun.run_function(
        'evaluate',
        handler="start_evaluate",
        params={},
        output=[]
    )

