Sarah Gross, student 2017280160


# Presentation

This project is Homework 3 of Machine Learning course of Tsinghua Univeristy.

## Requirements

* numpy (all)
* scipy (all)
* networkx (deepwalk, line, node2vec)
* gensim (deepwalk, node2vec)
* tensorflow (line)

To install the requirements, run ```pip install -r requirements.txt```

# Usage

## Embedding algorithms

1) Deepwalk

```python deepwalk/main.py --input cora/network.npz --output cora/cora.deepwalk.embeddings --num-walks 10 --walk-length 40 --model skipgram```

This program runs in approximately 15s for 10 walks of length 40. [paper] (https://arxiv.org/pdf/1403.6652)

Arguments :
* input : network file path (.npz file, run cora/data_utils_cora.py to get network.npz matrix)
* output : embedding output path
* num-walks : for each node, number of walks starting from this node  (optional)
* walk-length : size of each walk (number of vertices visited from each node)  (optional)
* model : model applied to each walk (word2vec or skipgram)  (optional)

2) LINE

```python line/main.py --input tencent/adj_train.npz --output tencent/tencent.line.embeddings --iter 500 --proximity second-order```

This program runs in approximately 35mins for 500 iterations (batches). [paper] (https://arxiv.org/pdf/1503.03578.pdf)

Arguments :
* input : network file path (.npz file)
* output : embedding output path
* iter : number of iterations (optional)
* proximity : depth for neighbour analysis (first-order or second-order) (optional)

3) Node2Vec

```python node2vec/main.py --input cora/network.npz --output cora/cora.node2vec.embeddings --num-walks 10 --walk-length 40```

This program runs in approximately 15s for 10 walks of length 40. [paper] (http://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)

Arguments :
* input : network file path (.npz file, run cora/data_utils_cora.py to get network.npz matrix)
* output : embedding output path
* num-walks : for each node, number of walks starting from this node  (optional)
* walk-length : size of each walk (number of vertices visited from each node)  (optional)

4) AANE

```python aane/main.py --input tencent/adj_train.npz --output tencent/tencent.aane.embeddings --iter 10```

This program runs in approximately 25 mins for 10 iterations. [paper] (http://www.public.asu.edu/~jundongl/paper/SDM17_AANE.pdf)

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


# Additional info

All algorithms an embedding dimension of size 128.

Algorithms relying on word2vec (deepwalk, node2vec) use 5 windows and as many workers as available cpus on the running computer.