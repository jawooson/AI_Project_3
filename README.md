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

## Intended Structure of documentation
1. Discuss BoW generally
2. Discuss BoW NN
3. Discuss BoW implementation in CodeSearchNet

4. Go into how the testing is done, metrics used, methodology. 
	a. How is testing is done?
	b. What is MRR, why does MRR need to be used here
	c. Discuss why leaderboard uses NDCG .


## 1. Some Notes on Bag of Words Model
* [https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/](https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/) This is a good article discussing generally what BoW is. 
* It really isn't a NN structure per say, it is a feature engineering pre processing step (BoW) that is then fed into a neural net. 
* The purpose of Bag of Words is to represent text data in a way that machine learning/AI algorithms can use. So, in the case for Neural Nets, BoW is meant to vectorize text data into a way that can be inputted as the input layer of a neural network. 
* This is interesting, because BoW has little to do with the neural net being used to model the data, it is simply a preprocessing step. The difficult part is knowing which deep learning model to use with this text data. Also, there are many ways to vectorize text data. 
* For BoW, we need to create a vocabulary, which is just all the unique words found in an entire corpus (collection of documents). This will create the vector of words/tokens. If the corpus contains n unique words, then the vector length will be n. A caveat, there are many heuristics into how we choose unique words. Often times we drop out words from our vocabulary that have low term frequency. This is because the value of n can be extremely large, so to limit the size of it we must drop some words.

 For each each document, we vectorize it. I provide a classic example on how vectorization is implemented. 

Given a corpus of 4 documents below:
1. _“It was the best of times”_
2. _“It was the worst of times”_  
3. _“It was the age of wisdom”_
4. _“It was the age of foolishness”_

Our vocabulary, the unique words in the corpus, is:
[_‘It’, ‘was’, ‘the’, ‘best’, ‘of’, ‘times’, ‘worst’, ‘age’, ‘wisdom’, ‘foolishness’_]

We vectorize each individual document by checking the frequency of each word from the n unique words that comprise our vocabulary. 

Rest of the documents will be:  
_“It was the best of times” = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  
“It was the worst of times” = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0]  
“It was the age of wisdom” = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0]  
“It was the age of foolishness” = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1]_

This is generally what Bag of Words is, it translates text data to a from that is ingestible to a neural net. 



## I need to discuss what exactly is being tested.
* Really unsure right now.
* [https://arxiv.org/pdf/1909.09436.pdf](https://arxiv.org/pdf/1909.09436.pdf) This is a paper regarding the project in general, might offer some good insight. 
* Ok so testing is interesting. The test set consists of 99 queries. For each query, we are given 1000 code snippets. Of the 1000 code snippets, only one is relevant and 999 are distractors, so the evaluation task is to rank them.  

## I'm going to need to discuss MRR and the other evaluation metrics used. I think this resource will be good. 
[https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)

### In test.py, MRR is used for test accuracy:
* Mean Reciprocal Rank is very simple. It measures where the first relevant term is. So, given CodeSearchNet, it measures where the one relevant code snippet is positioned relative to the other 999. Given its absolute rank, find the reciprocal. For example, if for one query the model positions in the 6th slot, it is computer as 1/6. Given that we have 99 different queries, MRR just takes the mean of all these reciprocal ranks. Again, for example if there are 3 queries with the first reciprocal rank given above and the other two are 1/8 and 1, the MRR is 1/3*(1/6 + 1/8 + 1). 

This metric used for accuracy is much better than traditional accuracy score because it deals with ranked data. Given a particular query and a test set, we want to find the most relevant code snippet compared. It is clear why rank is essential in testing because order is essential.  
[https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)


### In the W&B competition website, nDCG is used to rank different learning methods:
* Normalized Discounted Cumulative Gain 



[https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1](https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1)





## Bibliography

Research Papers Referenced and Used:
1. [Continuous Learning in Single-Incremental-Task Scenarios](https://arxiv.org/abs/1806.08568)
	* This paper describes Continual Learning, Single-Incremental-Task, New Classes problem, and catastrophic forgetting. They have a great description of the Naive, Rehearsal, and Elastic Weight Consolidation approach to solving Continual Learning. 
2. [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
	* This is the first paper to propose the Elastic Weight Consolidation approach to solving Continual Learning. 
3. [Compete to Compute](https://papers.nips.cc/paper/5059-compete-to-compute)
	* This paper describes how the order of your training data matters. 
4. [CORe50: a New Dataset and Benchmark for Continuous Object Recognition](http://proceedings.mlr.press/v78/lomonaco17a/lomonaco17a.pdf)
	* This paper describes the CORe50 dataset. In addition, the authors used the dataset to test several Continual Learning methods and compare their benchmarks. 
5. [Memory Efficient Experience Replay for Streaming Learning](https://arxiv.org/abs/1809.05922)


Datasets Used:  
* [CORe50 Dataset](https://vlomonaco.github.io/core50/)
	* The dataset we use. 

Code Used As a Starting Point: 
* [CVPR clvision challenge](https://github.com/vlomonaco/cvpr_clvision_challenge)
	* The starting point for the code we developed. This includes the loader for the CORe50 Dataset. Also, included is the Naive approach to continual learning that we use a baseline benchmark. 
* [Intro To Continual Learning](https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb)
	* Provided a model for the implementation of Naive, Rehearsal, and Elastic Weight Consolidation. We used this code in the development of our implementation. 
<!--stackedit_data:
eyJoaXN0b3J5IjpbODIxMzIyNjk0LDE5NjIzMzA1MjcsLTc4OD
gzMzc0MSwyMDE3MDE0Nzc5LDEzMzI4MDM3NzcsOTI4NTA4MDM3
LC0xMTg1MTE5NzE2LDEwODYwMzA2MjIsLTk2ODYyMzQ1OSwxOT
gzMzczODk5LC0xNjk1NDkwMTA3LC0zMzMyNTQ4OTIsLTE1NDI3
MzgyOTQsLTcxNjc2NjQ1NiwtMTM3MDc3MDk2NywxNzAwOTEwMD
g4LC02NjYxNzY1NDUsLTIwMjMzODE4ODVdfQ==
-->