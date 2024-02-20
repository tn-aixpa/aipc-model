# The two frequent culprits of model quality decay are 
# 1- data quality issues and 
# 2- changes in the input data distributions. 
import fasttext
from evidently import ColumnMapping

# prepare data and map schema
column_mapping = ColumnMapping()
column_mapping.target = "labels"
column_mapping.predictions = ""
column_mapping.text_features = ['text']
column_mapping.categorical_features = []


# model predictions
model_path = "../output-folder/allTokens_unfiltered_model.bin"
model = fasttext.load_model(model_path)



