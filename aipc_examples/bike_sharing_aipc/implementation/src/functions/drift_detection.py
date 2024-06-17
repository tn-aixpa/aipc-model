import mlrun
from functions.aipc_utils import load_validation_card_specifications

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, RegressionTestPreset
from evidently.tests import *

def model_performance_report(current):
    target = 'target'
    prediction = 'prediction'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']
    column_mapping = ColumnMapping()

    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features
    
    card_specifications = load_validation_card_specifications()
    for validation in card_specifications:
        validation_name = validation["name"]
        reference_dataset_path = validation["parameters"]["reference_model_performance_dataset"]
        reference_dataset = mlrun.get_dataitem(reference_dataset_path).as_df()
        print(reference_dataset)
        
        regression_performance = Report(metrics=[RegressionPreset()], 
                                    options={"render": {"raw_data": True}})
        regression_performance.run(current_data=current, reference_data=reference_dataset, column_mapping=column_mapping)
        regression_performance.save_html(f"reports/model_performance_{validation_name}_report.html")
        
        
        
def data_drift_report(current):
    card_specifications = load_validation_card_specifications()
    for validation in card_specifications:
        validation_name = validation["name"]
        reference_dataset_path = validation["parameters"]["reference_training_dataset"]
        print(validation)
        reference_dataset = mlrun.get_dataitem(reference_dataset_path).as_df()
        
        tests = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(),
        ])

        tests.run(reference_data=reference_dataset, current_data=current)
        tests.save_html(f"./data_drift_{validation_name}_report.html")