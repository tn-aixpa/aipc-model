
# Dataset Card for {{ pretty_name | default("Dataset Name", true) }}

## Dataset Description

- **Homepage:** {{ homepage_url | default("", true)}}
- **Repository:** {{ repo_url | default("", true)}}
- **Paper:** {{ paper_url | default("", true)}}
- **Leaderboard:** {{ leaderboard_url | default("", true)}}
- **Point of Contact:** {{ point_of_contact | default("", true)}}

### Dataset Summary

{{ dataset_summary | default("[More Information Needed]", true)}}

### Supported Tasks and Leaderboards

{{ supported_tasks_and_leaderboards_section | default("[More Information Needed]", true)}}

### Languages

{{ languages_section | default("[More Information Needed]", true)}}

## Dataset Structure

### Data Instances

The summarized documents will be formatted in a way that allows the end user to choose how much of the text they want to keep. The format is as follows:

"document_id": {
    "title": "document_title",
    "link": "document_link",
    "eurovoc_classifiers": [
        "classifier_1",
        "classifier_2",
        ...
    ],
    "full_text": [
        "sentence_1",
        "sentence_2",
        ...
    ],
    "importance": [
        0.7326374,
        0.1277499,
        ...
    ]
}

### Data Fields

{{ data_fields_section | default("[More Information Needed]", true)}}

### Data Splits

{{ data_splits_section | default("[More Information Needed]", true)}}

## Dataset Creation

### Curation Rationale

{{ curation_rationale_section | default("[More Information Needed]", true)}}

### Source Data

#### Initial Data Collection and Normalization

{{ data_collection_section | default("[More Information Needed]", true)}}

#### Who are the source language producers?

{{ source_language_producers_section | default("[More Information Needed]", true)}}

### Annotations

#### Annotation process

{{ annotation_process_section | default("[More Information Needed]", true)}}

#### Who are the annotators?

{{ who_are_annotators_section | default("[More Information Needed]", true)}}

### Personal and Sensitive Information

{{ personal_and_sensitive_information_section | default("[More Information Needed]", true)}}

## Considerations for Using the Data

### Social Impact of Dataset

{{ social_impact_section | default("[More Information Needed]", true)}}

### Discussion of Biases

{{ discussion_of_biases_section | default("[More Information Needed]", true)}}

### Other Known Limitations

{{ known_limitations_section | default("[More Information Needed]", true)}}

## Additional Information

### Dataset Curators

{{ dataset_curators_section | default("[More Information Needed]", true)}}

### Licensing Information

{{ licensing_information_section | default("[More Information Needed]", true)}}

### Citation Information

{{ citation_information_section | default("[More Information Needed]", true)}}

### Contributions

{{ contributions_section | default("[More Information Needed]", true)}}
