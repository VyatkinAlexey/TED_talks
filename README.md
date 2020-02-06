# TED talks classification

### Data
Data was collected from Kaggle, available by link: [data](https://www.kaggle.com/rounakbanik/ted-talks).

### Problem statement
Having 2550 records of TED talks, which consist of decription, full transcript and tags set build language model to predict 
**tags** by **description** and **transcript**.

### Project steps

#### 1. Tags clusterisation
There are **416** unique labels in the dataset, which consists of 2550 records. 
Thus it was necessary to reduce number of tags. As many of them are semantically close to each other, 
clustering was based on words embeddings.

Particularly, it was decided to use [glove](http://nlp.stanford.edu/data/glove.6B.zip) embedding,
which was taken from official [site](https://nlp.stanford.edu/projects/glove/).

Then Agglomerative clustering (to **7** clusters) was applied to tags embeddings. Clusters were named automatically by calculating nearest word to cluster mean (in terms of word embeddings). By manual check it was noticed that tags inside same cluster are related to one area. For example, __"disease"__ cluster has tags: addiction, autism spectrum disorder, blindness, cancer and so on.

In order to visualize the results [UMAP](https://arxiv.org/abs/1802.03426) dimensional reduction method was applied
to first **50** Principal Components of tag embedding matrix. Two-dimensional UMAP embedding with clusters labeled by color is shown below.

![clustering image](img/clustering.png)

Although 2 clusters (__that__ and __sense__, purple and green on the image) were too broad (thus there were named by very common word, which is in the "center" of embedding space), that is why they were eliminated from further analysis. 

Finally, we have **5** clusters and initial problem is to classify objects to 5 classes. Although, it should be noticed that classes are highly imbalanced, as shown on plot below.

![cluster balance](img/clustering_count.png)

#### 2. Feature extraction

In order to transform text into features, we applied TF-IDF method (excliding all English stopwords) 
to all transcripts of TED talks. The method basically matchs text with adjusted word frequency from this text. 
Despite strict filtering (each word should be found in text at least 4 times to be encountered) TF-IDF matrix is very sparse, as shown at plot.


