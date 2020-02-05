# TED talks classification

### Data
Data was collected from Kaggle, available by link: [data](https://www.kaggle.com/rounakbanik/ted-talks).

### Problem statement
Having 2550 records of TED talks, which consist of decription, full transcript and tags set build language model to predict 
**tags** by **description** and **transcript**.

### Project steps

#### 1. Tags clusterisation
There are 416 unique labels in the dataset, which consists of 2550 records. 
Thus it was necessary to reduce number of tags. As many of them are semantically close to each other, 
clustering was based on words embeddings.

Particularly, it was decided to use [glove](http://nlp.stanford.edu/data/glove.6B.zip) embedding,
which was taken from official [site](https://nlp.stanford.edu/projects/glove/).
