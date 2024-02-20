
# AI Product Card for {{ name of th final product }}

<!-- Provide a quick summary of what the produuct is. -->

## AI Product Card Details

### AI Product Description

<!-- Provide a longer summary of what this product is. -->

{{ product_name | default("", true) }}
{{ product_description | default("", true) }}

## List of AI Tasks

### Reference to AI task
 TODO
### Reference to Model Card

{{model_card_name}}

### Reference to Dataset Card

{{dataset_card_name}}

### Adaptation techniques

<!-- Provide a detailed description of the adaptation techniques used for the specific task. It may include the code implementation and also the data characteristis of deciding when to adapt the problem to different contexts-->

{{code_ implementation_to_adapt }}
{{data_profiling_for_deciding_when_to_adapt}}

### Optimisation tehniques

<!-- Provide a detailed descirption of the optimization techniques for the task. It may include techniques that improves the performance, precision or time of training/inference -->

{{code_implementation}}
 