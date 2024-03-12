import pandas as pd
import numpy as np
import requests
import zipfile
import io

from datetime import datetime, time
from sklearn import datasets, ensemble


def train():
    content = requests.get("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", verify=False).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday'], index_col='dteday')
    