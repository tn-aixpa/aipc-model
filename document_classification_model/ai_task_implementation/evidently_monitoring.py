import pandas as pd
import requests
import zipfile
import io
import random

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

!pip install googletrans==3.1.0a0

!pip install git+https://github.com/evidentlyai/evidently.git@main

# install evidently from master

try:
    import evidently
except:
    !pip install git+https://github.com/evidentlyai/evidently.git

from evidently.pipeline.column_mapping import ColumnMapping

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import ClassificationQualityMetric, TextDescriptorsDriftMetric, ColumnDriftMetric

"""# Load data

We will work with a dataset that contains reviews and ratings for different drugs.
"""

content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip").content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("drugsComTest_raw.tsv"), sep='\t')

raw_data = raw_data[['drugName', 'condition', 'review',	'rating']]

"""Data source: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29

Citation:
Felix Gräßer, Surya Kallumadi, Hagen Malberg, and Sebastian Zaunseder. 2018. Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning. In Proceedings of the 2018 International Conference on Digital Health (DH '18). ACM, New York, NY, USA, 121-125. DOI: [Web Link]
"""

raw_data.head()

"""# Training and testing the model

Suppose we want to build a model to distinguish between reviews with rating 1 (negative review) and 10 (positive review). Let's also assume that we only have access to reviews on pain medications.
"""

init_data = raw_data.loc[(raw_data['condition'] == 'Pain') & (raw_data['rating'].isin([1, 10])), ['review', 'rating']]
init_data['is_positive'] = init_data['rating'].apply(lambda x: 0 if x == 1 else 1)
init_data.drop(['rating'], inplace=True, axis=1)
init_data.head()

"""We split the data into "reference" and "valid" datasets. Reference dataset is used for training while 40% of the data is held out for model validation"""

X_train, X_test, y_train, y_test = train_test_split(init_data['review'], init_data['is_positive'],
                                                    test_size=0.4, random_state=42, shuffle=True)

reference = pd.DataFrame({'review': X_train, 'is_positive': y_train})
valid = pd.DataFrame({'review': X_test, 'is_positive': y_test})

"""Train a model with TF-IDF vectorization and linear classifier on top"""

pipeline = Pipeline(
    [
        ("vectorization", TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english")),
        ("classification", SGDClassifier(alpha=0.0001, max_iter=50, penalty='l1', loss='modified_huber', random_state=42))
        ])
pipeline.fit(reference['review'].values, reference['is_positive'].values)

"""Calculate model predictions for training and validation datasets. Our model predicts the probability of a review being positive"""

reference['predict_proba'] = pipeline.predict_proba(reference['review'].values)[:,1]
valid['predict_proba'] = pipeline.predict_proba(valid['review'].values)[:,1]

# set up column mapping
column_mapping = ColumnMapping()

column_mapping.target = 'is_positive'
column_mapping.prediction = 'predict_proba'
column_mapping.text_features = ['review']

# list features so text field is not treated as a regular feature
column_mapping.numerical_features = []
column_mapping.categorical_features = []

"""Model accuracy on validation dataset is a bit higher than 0.8. This is the level of performance we can expect on similar new data"""

performance_report = Report(metrics=[
    ClassificationQualityMetric()
])

performance_report.run(reference_data=reference, current_data=valid,
                        column_mapping=column_mapping)
performance_report

"""# Data drift due to "technical issues"

Imagine that after deploying the model something changes at the data collecting stage. Could be that on-line data preprocessing differs from historical one or that some odd update simply breaks data cleaning steps.

Let's see what happens if we either inject some random html tags in the review text or translate the review to French.
"""

from googletrans import Translator
translator = Translator()

def translate_str(s):
  return translator.translate(s, dest='fr').text

random_html_tags = ('<body>, </body>', '<html><body>', '</body></html>', '<h1>', '</h1>',
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 0 0" width="0" height="0" focusable="false" role="none" style="visibility: hidden; position: absolute; left: -9999px; overflow: hidden;"><defs><filter id="wp-duotone-magenta-yellow"><feColorMatrix color-interpolation-filters="sRGB" type="matrix" values=" .299 .587 .114 0 0 .299 .587 .114 0 0 .299 .587 .114 0 0 .299 .587 .114 0 0 "></feColorMatrix><feComponentTransfer color-interpolation-filters="sRGB"><feFuncR type="table" tableValues="0.78039215686275 1"></feFuncR><feFuncG type="table" tableValues="0 0.94901960784314"></feFuncG><feFuncB type="table" tableValues="0.35294117647059 0.47058823529412"></feFuncB><feFuncA type="table" tableValues="1 1"></feFuncA></feComponentTransfer><feComposite in2="SourceGraphic" operator="in"></feComposite></filter></defs></svg>')

def inject_random_html_tags(s):
  num_tags = 25
  for i in range(num_tags):
    random.seed(i)
    pos = random.choice(range(len(s)))
    s = s[:pos] + random.choice(random_html_tags) + s[pos:]

  return s

valid_disturbed = valid[['review', 'is_positive']]

disturbed_num = int(len(valid_disturbed) * 0.5)
random.seed(42)
disturbed_ind = random.sample(list(valid_disturbed.index), k=disturbed_num)
valid_disturbed.loc[disturbed_ind[:int(disturbed_num / 10)], 'review'] = \
valid_disturbed.loc[disturbed_ind[:int(disturbed_num / 10)], 'review'].apply(inject_random_html_tags)
valid_disturbed.loc[disturbed_ind[int(disturbed_num / 10):], 'review'] = \
valid_disturbed.loc[disturbed_ind[int(disturbed_num / 10):], 'review'].apply(translate_str)

valid_disturbed['predict_proba'] = pipeline.predict_proba(valid_disturbed['review'].values)[:,1]

performance_report = Report(metrics=[
    ClassificationQualityMetric()
])

performance_report.run(reference_data=valid, current_data=valid_disturbed,
                        column_mapping=column_mapping)
performance_report

"""Oops! Model accuracy has dropped. Let's look at the Data Drift report to see why"""

import nltk
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')

data_drift_report = Report(
    metrics=[
        ColumnDriftMetric('is_positive'),
        ColumnDriftMetric('predict_proba'),
        TextDescriptorsDriftMetric(column_name='review'),
    ]
)

data_drift_report.run(reference_data=reference,
                      current_data=valid_disturbed,
                      column_mapping=column_mapping)
data_drift_report

"""Here we see the culprit: new (perturbed) dataset contains considerably more suspiciously long reviews and reviews with a lot of OOV (out-of-vocabulary) words

If we look at the examples of such reviews we see the problems right away:

*   HTML tags not being removed from the texts properly
*   Reviews in a new unexpected language


"""

from evidently.features.text_length_feature import TextLength
from evidently.features.OOV_words_percentage_feature import OOVWordsPercentage

text_feature = TextLength(column_name='review').generate_feature(data=valid_disturbed, data_definition=None)
oov_feature = OOVWordsPercentage(column_name='review').generate_feature(data=valid_disturbed, data_definition=None)

valid_disturbed['text_length'] = text_feature.values
valid_disturbed['oov_share'] = oov_feature.values

valid_disturbed[valid_disturbed['text_length'] > 1000].head()

valid_disturbed[valid_disturbed['text_length'] > 1000].iloc[0, 0]

valid_disturbed[valid_disturbed['oov_share'] > 30].head()

data_drift_dataset_report = Report(metrics=[
    ColumnDriftMetric(column_name='review')
])

data_drift_dataset_report.run(reference_data=reference,
                              current_data=valid_disturbed,
                              column_mapping=column_mapping)
data_drift_dataset_report

"""# Content drift

Technical issues sorted out, the model continues to be used for reviews' sentiment analysis. Suppose we decide to apply it on reviews for antidepressants.
"""

new_content = raw_data.loc[(raw_data['condition'] == 'Depression') & (raw_data['rating'].isin([1, 10])), ['review', 'rating']]
new_content['is_positive'] = new_content['rating'].apply(lambda x: 0 if x == 1 else 1)
new_content.drop(['rating'], inplace=True, axis=1)
new_content.head()

new_content['predict_proba'] = pipeline.predict_proba(new_content['review'].values)[:,1]

performance_report = Report(metrics=[
    ClassificationQualityMetric(),
])

performance_report.run(reference_data=valid, current_data=new_content,
                        column_mapping=column_mapping)
performance_report

"""Unfortunately, model's performance is worse than expected. Let's look at the Data Drift report"""

data_drift_report = Report(
    metrics=[
        ColumnDriftMetric('is_positive'),
        ColumnDriftMetric('predict_proba'),
        TextDescriptorsDriftMetric(column_name='review'),
    ]
)

data_drift_report.run(reference_data=reference,
                      current_data=new_content,
                      column_mapping=column_mapping)
data_drift_report

"""We see that there's drift in the data. Reviews tend to be longer for the current dataset and OOV words are encountered more often. But nothing as obvious as in the case above.

The problem is that it's the reviews *content* that drifted. Let's see how Evidently can help to detect such a change

To detect content drift Evidently uses domain classifier approach. A classifier is trained that tries to predict whether a text is from a reference dataset or from a new dataset. If it can be done successfully than the new dataset is significantly different from the reference one.

If content data drift is detected Evidently also provides some insights on the nature of the drift:
* Words that are more distinctive of the current vs reference dataset.
These are the words that are the most informative for the domain classifier when it predicts if a text came from the reference or from the current dataset
* Examples of texts that are more distinctive of the current vs reference dataset. These examples were the easiest for a classifier to label correctly
"""

data_drift_dataset_report = Report(metrics=[
    ColumnDriftMetric(column_name='review')
])

data_drift_dataset_report.run(reference_data=reference,
                              current_data=new_content,
                              column_mapping=column_mapping)
data_drift_dataset_report

"""At once we can see how the current dataset differs from the reference dataset our model was trained on. Current dataset is characterized with words and examples about depression, mood and popular antidepressants while the reference dataset is more about pain, shock symptoms and popular painkillers

Note that no such drift is detected for validation dataset that consists of reviews for painkillers, similar to reference dataset
"""

data_drift_dataset_report = Report(metrics=[
    ColumnDriftMetric(column_name='review')
])

data_drift_dataset_report.run(reference_data=reference,
                              current_data=valid,
                              column_mapping=column_mapping)
data_drift_dataset_report

"""One of the solutions to deal with this kind of data change is to retrain the model on a dataset that includes new relevant data. With Evidently it can be done *proactively* by detecting data drift even before information on target labels and model performance is collected"""