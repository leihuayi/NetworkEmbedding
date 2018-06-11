Sarah Gross, student 2017280160


# Presentation

This project is Homework 3 of Machine Learning course of Tsinghua Univeristy.

## Requirements

numpy>=1.12.1
scipy>1.12.1
argparse>=1.2.1
futures==3.1.1
six>=1.7.3
gensim>=1.0.0
psutil>=2.1.1
networkx
tensorflow>=1.3.0

To install the requirements, run ```pip install -r requirements.txt```

# Usage

## Embedding algorithms

1) Deepwalk

```python deepwalk/main.py --input cora/network.npz --output cora/cora.deepwalk.embeddings --number-walks 10 --walk-length 40 --model skipgram```

This program runs in approximately 15 mins for 20 walks of length 20.

Arguments :
* input : network file path (.npz file, run cora/data_utils_cora.py to get network.npz matrix)
* output : embedding output path
* number-walk : for each node, number of walks starting from this node  (optional)
* walk-length : size of each walk (number of vertices visited from each node)  (optional)
* model : model applied to each walk (word2vec or skipgram)  (optional)

2) LINE

```python line/main.py --input tencent/adj_train.npz --output tencent/tencent.line.embeddings --proximity second-order --iter 1000```

This program runs in approximately one hour for 1000 iterations (batches).

Arguments :
* input : network file path (.npz file)
* output : embedding output path
* proximity : depth for neighbour analysis (first-order or second-order) (optional)
* iter : number of iterations (optional)

3) Node2Vec

```python node2vec/main.py --input cora/cora.mat --output cora/cora.deepwalk.embeddings --number-walks 10 --walk-length 40```

Arguments :
* input : network file path (.npz file, run cora/data_utils_cora.py to get network.npz matrix)
* output : embedding output path
* number-walk : for each node, number of walks starting from this node  (optional)
* walk-length : size of each walk (number of vertices visited from each node)  (optional)

4) AANE

This program runs in approximately 30 mins for 10 iterations.

```python aane/main.py --input tencent/adj_train.npz --output tencent/tencent.aane.embeddings --iter 10```

Arguments :
* input : network file path (.npz file)
* output : embedding output path
* iter : number of iterations (optional)


## Evaluation

1) Dataset Cora

```python evaluate_cora.py  --emb cora/cora.algo.embeddings  --net cora/network.npz --labels cora/labels.npz```

Arguments :
* emb : embeddings file path
* net : network file path (.npz file, run cora/data_utils_cora.py to get network.npz matrix)
* labels : y sparse matrix (.npz file, run cora/data_utils_cora.py to get labels.npz matrix)

2) Dataset Tencent Weibo

```python evaluate_tencent.py --emb tencent/tencent.algo.embeddings --net tencent/adj_train.npz --testdir tencent```

Arguments :
* emb : embeddings file path
* net : network file path (.npz file)
* testdir : directory containing test_edges.npy and test_edges_false.npy