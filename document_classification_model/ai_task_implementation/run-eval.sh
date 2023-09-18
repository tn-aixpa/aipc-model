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
  python scripts/06-micro_macro.py $FOLDER/${type}_unfiltered.results.txt $FOLDER/${type}_unfiltered.test.txt
  for b in "${bys[@]}"; do
    echo $b
    python scripts/06-micro_macro.py $FOLDER/${type}_${b}_filtered.results.txt $FOLDER/${type}_${b}_filtered.test.txt
  done
done
