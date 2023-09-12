
# AI Product Card for Text Classification

{{ Provide a quick summary of what the produuct is. }}


## Reference to Domain Card

## AI Product Card Details

The primary focus of this Product Card is to develop an automatic pipeline for the classification of the Gazzetta Ufficiale documents. </br>


### AI Product Description

{{Provide a longer summary of what this product is.}}

- **What is the product name:** Italian legislative text classification for Gazzetta Ufficiale
- **Provide a long and detailed description of the product:** From a machine learning perspective, this text classification task is a multi-class, multi-label problem, i.e. each target document (the act’s title in this case) is to be labeled with one or more classes from the subject index.

## List of AI Tasks

 
### Reference to Model Card

{{model_card_name}}

### Reference to Dataset Card

{{dataset_card_name}}

### Adaptation techniques

{{ Provide a detailed description of the adaptation techniques used for the specific task. It may include the code implementation and also the data characteristis of deciding when to adapt the problem to different contexts }}

- **Provide code implementation** {{code_ implementation_to_adapt }}
- **Provide data profiling for deciding when to adapt the solution** {{data_profiling_for_deciding_when_to_adapt}}

### Optimisation techniques

- **Provide detailed description of the optimization technique** {{description}} Provide a detailed description of the optimization techniques for the task. </br>
It may include techniques that improves the performance, precision or time of training/inference

- **Provide code implementation **{{code_implementation}}
#### preprocessing the data </br>
Looking at the data, we observe that some words contain uppercase letter or punctuation. One of the first step to improve the performance of our model is to apply some simple pre-processing.
</br>
A crude normalization can be obtained using command line tools such as sed and tr
```
>> cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
>> head -n 12404 cooking.preprocessed.txt > cooking.train
>> tail -n 3000 cooking.preprocessed.txt > cooking.valid
```

Let's train a new model on the pre-processed data: </br>
```
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train")
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 82041  lr: 0.000000  loss: 5.671649  eta: 0h0m

>>> model.test("cooking.valid")
(3000L, 0.164, 0.0717)
```

#### more epochs and larger learning rate

By default, fastText sees each training example only five times during training, which is pretty small, given that our training set only have 12k training examples. The number of times each examples is seen (also known as the number of epochs), can be increased using the -epoch option: 
```
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train", epoch=25)
Read 0M words
Number of words:  9012
Number of labels: 734
Progress: 100.0%  words/sec/thread: 77633  lr: 0.000000  loss: 7.147976  eta: 0h0m
```