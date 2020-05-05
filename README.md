# NYU // AI // Project 03 // CodeSearchNet

This is an extension of the [CodeSearchNet](https://app.wandb.ai/github/codesearchnet/benchmark) competition. 

## A brief overview of the project
[CodeSearchNet](https://arxiv.org/abs/1909.09436)  is a collection of datasets and benchmarks that explore the problem of code retrieval using natural language. This research is a continuation of some ideas presented in this  [blog post](https://githubengineering.com/towards-natural-language-semantic-code-search/)  and is a joint collaboration between GitHub and the  [Deep Program Understanding](https://www.microsoft.com/en-us/research/project/program/)  group at  [Microsoft Research - Cambridge](https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/). We aim to provide a platform for community research on semantic code search via the following:

1.  Instructions for obtaining large corpora of relevant data
2.  Open source code for a range of baseline models, along with pre-trained weights
3.  Baseline evaluation metrics and utilities
4.  Mechanisms to track progress on a  [shared community benchmark](https://app.wandb.ai/github/CodeSearchNet/benchmark)  hosted by  [Weights & Biases](https://www.wandb.com/)

We hope that CodeSearchNet is a step towards engaging with the broader machine learning and NLP community regarding the relationship between source code and natural language. We describe a specific task here, but we expect and welcome other uses of our dataset.

More context regarding the motivation for this problem is in this  [technical report](https://arxiv.org/abs/1909.09436).

## Structure of Documentation
1. Bag of Words Model
2. Neural Bag of Words
3. CodeSearchNet Baseline Model: Neural Bag of Words

4. Testing  
	a. How does CodeSearchNet Implement Testing?  
	b. Metrics for Test Accuracy: MRR  
	c. Metrics for Test Accuracy: nDCG  

5. Pre-trained model results

## 1. Bag of Words Model

* The Bag of Words (BoW) model does not do prediction. It is a feature engineering pre processing step.
* The purpose of BoW is to represent text data in a way that machine learning/AI algorithms can use. So, in the case for Neural Nets, BoW is meant to vectorize text data into a way that can be inputted as the input layer of a neural network. 
* The difficult part is knowing which deep learning model to use with this text data. Also, there are many ways to vectorize text data. 
* For BoW, we need to create a vocabulary, which is just all the unique words found in an entire corpus (collection of documents). This will create the vector of words/tokens. If the corpus contains n unique words, then the vector length will be n. A caveat, there are many heuristics into how we choose unique words. Often times we drop out words from our vocabulary that have low term frequency. This is because the value of n can be extremely large, so to limit the size of it we must drop some words.

For each each document, we vectorize it. I use an example derived by [Jocelyn D'Souza](https://medium.com/@djocz) in his Medium article [An Introduction to Bag-of-Words in NLP](https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428).

Given a corpus of 4 documents below:
1. _“It was the best of times”_
2. _“It was the worst of times”_  
3. _“It was the age of wisdom”_
4. _“It was the age of foolishness”_

Our vocabulary, the unique words in the corpus, :
[_‘It’, ‘was’, ‘the’, ‘best’, ‘of’, ‘times’, ‘worst’, ‘age’, ‘wisdom’, ‘foolishness’_]

These unique words that comprise the vocabulary are commonly known as tokens. 

We vectorize each individual document by checking the frequency of each word from the n unique words that comprise our vocabulary. 

Rest of the documents will be:  
_“It was the best of times” = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  
“It was the worst of times” = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0]  
“It was the age of wisdom” = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0]  
“It was the age of foolishness” = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1]_  
 [[8]](https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428) .

This is generally what Bag of Words is, it translates text data to a from that is ingestible to a neural net. 

While this idea sounds very simple, there are many different methodologies in how to vectorize text data, such as dropping out low frequency words, using different metrics in the vectorized notation such as term frequency or inverse document frequency.


## 2. Neural Bag of Words
The NBOW model takes an average of the word vectors in the input text and performs classification with a logistic regression layer. Essentially the NBOW model is a fully connected feed forward network with BOW input. [[2]](https://www.aclweb.org/anthology/W16-1626.pdf)


## 3. CodeSearchNet Baseline Model: Neural Bag of Words
The Neural Bag of Words baseline model used by CodeSearchNet is interesting because it only uses Bag of Words to create tokens and then those tokens are fed into a learnable embedding to create the vector representation. 

Word embedding differs from bag of words. Bag of words suffers from high dimensionality and sparsity. The total number of unique words comprising the vocabulary can be huge, and given any particular document, its vector representation can be mostly zeros. This is not good for neural networks, so word embedding is a different technique that aims to reduce the dimensionality of a words representation. 

"In an embedding, words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space.

The position of a word within the vector space is learned from text and is based on the words that surround the word when it is used." [[9]](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)

The position of a word in the learned vector space is referred to as its embedding.

<div align="center"><img src="https://github.com/jawooson/AI_Project_3/blob/jason-dev/images/figure_3.png" width=75%/></div>

 Figure taken from [[1]](https://arxiv.org/pdf/1909.09436.pdf)


In the Neural Bag of Words Model, the code to be evaluated and the NLP query are made into tokens. This is the bag of words stage in the nBoW model. After tokens are made, they are fed into a word embedder (sequence encoder). This creates vector representations, of the code and the query, in a predefined vector space (ex R^4). Finally, the distance is measured between the code and the query, which creates the ranking from closest to furthest. The default distance formula used is the cosine distance.

Cosine distance is commonly used in tasks utilizing text data because Euclidean distance can be skewed based on document sizes. Cosine distance is a more appropriate measurement because the angle of vectors is measured, which makes it a more robust distance measurement. [[12]](https://www.machinelearningplus.com/nlp/cosine-similarity/)

## 4a. How does CodeSearchNet Implement Testing?  
The test set consists of 99 queries. For each query, we are given 1000 code snippets. Of the 1000 code snippets, only one is relevant and 999 are distractors, so the evaluation task is to rank them.  [[1]](https://arxiv.org/pdf/1909.09436.pdf) 

## 4b. Metrics for Test Accuracy: MRR  
Mean Reciprocal Rank is very simple. It measures where the first relevant term is. So, given CodeSearchNet, it measures where the one relevant code snippet is positioned relative to the other 999. Given its absolute rank, find the reciprocal. For example, if for one query the model positions in the 6th slot, it is computer as 1/6. Given that we have 99 different queries, MRR just takes the mean of all these reciprocal ranks. Again, for example if there are 3 queries with the first reciprocal rank given above and the other two are 1/8 and 1, the MRR is 1/3*(1/6 + 1/8 + 1). 

This metric used for accuracy is much better than traditional accuracy score because it deals with ranked data. Given a particular query and a test set, we want to find the most relevant code snippet compared. It is clear why rank is essential in testing because order is essential.
[[13]](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)


## 4c. Metrics for Test Accuracy: nDCG
* Normalized Discounted Cumulative Gain is used for the W&B rankings because it takes into account different users running different models. I won't discuss the derivation of nDCG too heavily, but it is good at capturing the ranking of relevant documents, as well as varying number of test documents. 

#### Add picture found in AI/Assignment_3/readme_images/ndcg_diagram.png
[[13]](https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1)




## Bibliography

**Research Papers Referenced and Used:**  
[1] [CodeSearchNet Challenge Evaluating the State of Semantic Code Search](https://arxiv.org/pdf/1909.09436.pdf) 
* This is the academic paper that is associated by the creators of the CodeSearchNet Challenge. The paper goes into more detail regarding testing and how each baseline model is used. 

[2] [Learning Word Importance with the Neural Bag-of-Words Model](https://www.aclweb.org/anthology/W16-1626.pdf) 

[3] [Measuring Similarity of Academic Articles with Semantic Profile and Joint Word Embedding](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8195345)

**Main challenge github:**  
[4] [CodeSearchNet Github](https://github.com/github/CodeSearchNet/tree/e792e1caea20fbd4fba439565fe20c10d4798435)

**Challenge W&B Page:**  
[5] [CodeSearchNet W&B](https://app.wandb.ai/github/codesearchnet/benchmark)

**Online Resources cited and used:**  
[6] [How to Develop a Deep Learning Bag-of-Words Model for Sentiment Analysis (Text Classification)](https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/)  
[7] [A Gentle Introduction to the Bag-of-Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)  
[8] [An Introduction to Bag-of-Words in NLP](https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428)  
[9] [How to Use Word Embedding Layers for Deep Learning with Keras](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)  
[10] [Wikipedia: Word embedding](https://en.wikipedia.org/wiki/Word_embedding)  
[11] [Tensorflow Documentation: Word embeddings](https://www.tensorflow.org/tutorials/text/word_embeddings)  
[12] [Cosine Similarity – Understanding the math and how it works (with python codes)](https://www.machinelearningplus.com/nlp/cosine-similarity/)  
[13] [MRR vs MAP vs NDCG: Rank-Aware Evaluation Metrics And When To Use Them](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)  



 

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTUwOTM0MDE4NywxMjQ3NzUxMTEyLDU4Mz
U0MDQ1MywtMTE0Njk0ODQ3MywtMTc2ODY1NjQyOCwxMTA2NTY3
NTU5LDExMzM0MTg4NzYsOTYwODQ4OTY4LDE3ODY0MDk5OTksMT
ExNzY5MzQ4NiwxNDQxNTY2NTA5LDIwNzI3NzM1MywtNTg2NTMw
NjcyLC0xMTc2MjQ4MjM1LDEzOTM4OTc4NiwxNTUyMTM2NjksMT
c5MDcxMDI3Miw5NjA1NzgxNDYsMTI2NTY4Njg2Myw5MTk1ODIw
NDddfQ==
-->