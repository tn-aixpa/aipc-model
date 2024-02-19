# run data quality checks with evidently
# This file contains various evaluation metrics and tests and helps generate interactive reports for different scenarios. 
# In this case, we will create a custom report by combining several evaluations we want to run to understand data changes.   
import pandas as pd
import json
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import TextDescriptorsDriftMetric

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

from evidently.features.text_length_feature import TextLength
from evidently.features.OOV_words_percentage_feature import OOVWordsPercentage


# prepare data and map schema
column_mapping = ColumnMapping()
column_mapping.target = "labels"
#column_mapping.predictions = "predicted_labels"
column_mapping.text_features = ['text']
column_mapping.categorical_features = []
column_mapping.numerical_features = []

def load_data(path):
    with open(path, "r") as file:
        data = json.load(file)
        return data
text_data = load_data("../output-folder/data.json")
data = pd.read_json("../output-folder/data.json")
print(data)

# the reference dataset
reference = data[data['id'].str.startswith('ipzs-2021')]
# the currect dataset to be validated
current = data[data['id'].str.startswith('ipzs-2020')]


# report presets
report = Report(metrics=[
    DataDriftPreset(),
    TextDescriptorsDriftMetric(column_name="text")
    ])
report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
report.save_html("prova.html")
#test = report.as_dict()


def text_features():
    text_feature = TextLength(column_name='text').generate_feature(data=current, data_definition=None)
    oov_feature = OOVWordsPercentage(column_name='text').generate_feature(data=current, data_definition=None)

# test suites 
tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
])

tests.run(reference_data=reference, current_data=current)
print(tests)

