import huggingface_hub as hf
from huggingface_hub import ModelFilter
import datetime

models = hf.list_models(
    filter=ModelFilter(
		task="text-classification",
		library="pytorch",
		#model_name="bert"
	)
)
datasets = hf.list_datasets()
#last_date = datetime.datetime(2024, 1, 1, 16, 27, 24, tzinfo=datetime.timezone.utc)
#print(last_date)

for model in models:
    print(model)
    card = hf.hf_hub_download(model.modelId, 'README.md', local_dir=f"{model.modelId}") 
    
