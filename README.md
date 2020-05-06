# Test of Named Entity Recognition 

## Description

This test is based on example from [Transformers](https://github.com/huggingface/transformers/tree/master/examples/ner) library owned and maintained by [HuggingFace](https://huggingface.co/).

## Setup

Download all Python3 dependencies:

```sh
pip3 install virtualenv
virtualenv venv
# activate your virtual environment
pip3 install -r requirements.txt
```

Download all the data:

```sh
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > ./data/train.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-dev.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > ./data/dev.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-test.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > ./data/test.txt.tmp
```

Export few variables to be used by preprocessing script:

```sh
export MAX_LENGTH=64
export BERT_MODEL=bert-base-multilingual-cased
```

Run preprocessing commands:

```sh
python ./data/preprocess.py ./data/dev.txt.tmp $BERT_MODEL $MAX_LENGTH > ./data/dev.txt
python ./data/preprocess.py ./data/train.txt.tmp $BERT_MODEL $MAX_LENGTH > ./data/train.txt
python ./data/preprocess.py ./data/test.txt.tmp $BERT_MODEL $MAX_LENGTH > ./data/test.txt
```

Setup list of labels:

```sh
cat ./data/train.txt ./data/dev.txt ./data/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
```

## Run training

```sh
python main.go
```

## Tensorboard - see training results on graphs

```sh
tensorboard --logdir lightning_logs
```
