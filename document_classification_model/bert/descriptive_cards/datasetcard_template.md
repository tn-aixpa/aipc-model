---


{{ card_data }}
---

# Dataset Card for {{ pretty_name | default("Dataset Name", true) }}

## Dataset Description

- **Homepage:** {{ homepage_url | default("", true)}}
- **Repository:** {{ repo_url | default("", true)}}
- **Paper:** {{ paper_url | default("", true)}}
- **Leaderboard:** {{ leaderboard_url | default("", true)}}
- **Point of Contact:** {{ point_of_contact | default("", true)}}

### Dataset Summary

The General Series includes all acts like: 
ordinary laws 
presidential decrees, 
ministerial decrees 
resolutions, 
other regulatory acts from the central and peripheral state administrations.



### Supported Tasks and Leaderboards

{{ supported_tasks_and_leaderboards_section | default("[More Information Needed]", true)}}

### Languages

{{ languages_section | default("[More Information Needed]", true)}}

## Dataset Structure

### Data Instances

Example of one single data instance:
```
{
        "text": "Example of the text",
        "id": "ipzs-20210604_21G00088",
        "labels": [
            "A1810"
        ]
    }
```
### Data Fields

The data fields the model accepts are the following:
- text
- id
- labels

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
