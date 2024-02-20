
# Validation Card for {{ pretty_name | default("Dataset Name", true) }}

## Dataset Description

- **Homepage:** {{ homepage_url | default("", true)}}
- **Repository:** {{ repo_url | default("", true)}}
- **Paper:** {{ paper_url | default("", true)}}
- **Leaderboard:** {{ leaderboard_url | default("", true)}}
- **Point of Contact:** {{ point_of_contact | default("", true)}}

## Data Drift

Monitoring data drift is important when working with embeddings. 
When the legal document classification application is in production, it faces real-world data and we can detect when the data become too different or unusual and react before it affects the model quality. As the environment changes, this data might differ from what the model has seen during training. As a result, the so-called data drift can make the model less accurate. 

There are a few ways to evaluate the similarity between text datasets. 
One way is to compare the descriptive statistics of the text data (such as length of text, the share of out-of-vocabulary words, and the share of non-letter symbols) and explore if they have shifted between the two datasets. 
This option is available in Evidently as the Text Descriptors Drift metric. 
We will include it in the combined report in addition to evaluating drift in the model predictions and target.

### Embedding Drift detection

{{ languages_section | default("[More Information Needed]", true)}}

### Track the size of drift

To track the “size” of drift in time, you can pick metrics like Euclidean distance. However, you might need a few experiments to tune the alert thresholds since the values are absolute.


## Model Drift

{{ supported_tasks_and_leaderboards_section | default("[More Information Needed]", true)}}





