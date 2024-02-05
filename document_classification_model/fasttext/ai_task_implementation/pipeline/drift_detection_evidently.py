import pandas as pd

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import * 


def build_drift_detection_test(reference, current):
    tests = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestNumberOfDriftedColumns(),
    ])

    tests.run(reference_data=reference, current_data=current)
    tests_result = tests.as_dict()
    return tests_result['summary']['all_passed']



def build_data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping,
    drift_share=0.4,
) -> Report:
    """
    Returns a list with pairs (feature_name, drift_score)
    Drift Score depends on the selected statistical test or distance and the threshold
    """
    data_drift_report = Report(metrics=[DataDriftPreset(drift_share=drift_share)])
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    return data_drift_report
