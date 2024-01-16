# run data drifting checks with evidently

import io
import numpy as np
import os
import pandas as pd
from pathlib import *
import requests
import zipfile

from datetime import datetime
from sklearn import datasets, ensemble

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import ColumnDriftMetric
from evidently.metrics import TextDescriptorsDriftMetric

data_drift_report = Report(
    metrics=[
        ColumnDriftMetric('labels'),
        ColumnDriftMetric('predicted_labels'),
        TextDescriptorsDriftMetric(column_name='text'),
    ]
)

data_drift_report.run(reference_data=reference, 
                      current_data=valid_disturbed, 
                      column_mapping=column_mapping)
data_drift_report