#!/bin/bash

types=(goodTokens allLemmas allTokens)
# types=(goodTokens)
bys=(by_document by_label)

if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi

FOLDER=$1

for type in "${types[@]}"; do
  echo $type
  ./fastText-0.9.2/fasttext supervised -input $FOLDER/${type}_unfiltered.train.txt -output $FOLDER/${type}_unfiltered_model -epoch 25 -lr 1.0
  ./fastText-0.9.2/fasttext test $FOLDER/${type}_unfiltered_model.bin $FOLDER/${type}_unfiltered.test.txt
  ./fastText-0.9.2/fasttext predict $FOLDER/${type}_unfiltered_model.bin $FOLDER/${type}_unfiltered.test.txt > $FOLDER/${type}_unfiltered.results.txt
  python 06-micro_macro.py $FOLDER/${type}_unfiltered.results.txt $FOLDER/${type}_unfiltered.test.txt
  for b in "${bys[@]}"; do
    echo $b
    ./fastText-0.9.2/fasttext supervised -input $FOLDER/${type}_${b}_filtered.train.txt -output $FOLDER/${type}_${b}_filtered_model -epoch 25 -lr 1.0
    ./fastText-0.9.2/fasttext test $FOLDER/${type}_${b}_filtered_model.bin $FOLDER/${type}_${b}_filtered.test.txt
    ./fastText-0.9.2/fasttext predict $FOLDER/${type}_${b}_filtered_model.bin $FOLDER/${type}_${b}_filtered.test.txt > $FOLDER/${type}_${b}_filtered.results.txt
    python 06-micro_macro.py $FOLDER/${type}_${b}_filtered.results.txt $FOLDER/${type}_${b}_filtered.test.txt
  done
done
