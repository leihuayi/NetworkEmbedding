Sarah Gross, student 2017280160


# Presentation

This project is Homework 3 of Machine Learning course of Tsinghua Univeristy.

## Requirements

wheel>=0.23.0
Cython>=0.20.2
argparse>=1.2.1
futures==3.1.1
six>=1.7.3
gensim>=1.0.0
scipy>=0.15.0
psutil>=2.1.1


To install the requirements, run ```pip install -r requirements.txt```


# Usage

## Embedding algorithms

1) Deepwalk

```python deepwalk/main.py --format mat --input cora/cora.mat --number-walks 20 --walk-length 20 --model Word2Vec --output cora/cora.embeddings```

Arguments :
- format : graph format (edgelist / mat / adjlist)
- input : graph file path
- output : embedding output path
- number-walk : for each node, number of walks starting from this node
- walk-length : size of each walk (number of vertices visited from each node)
- model : model applied to each walk (Word2Vec / Skipgram)

2) LINE

3)

4)


## Evaluation

1) Cora

```python evaluate.py  --emb cora/cora.embeddings  --network cora/cora.mat```

2) Tencent Weibo