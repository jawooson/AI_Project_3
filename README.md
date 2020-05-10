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
1. Neural Bag of Words Model  
	a. Bag of Words Model  
	b. Neural Bag of Words  
	c. CodeSearchNet Baseline Model: Neural Bag of Words  
	
2. Bidirectional RNNs model  
 	a. Basic RNNs model  
	b. Bidirectional RNNs model  
	c. Gated Recurrent Unit  
	d. CodeSearchNet Baseline Model: Bidirectional RNN model (with GRU)  
	
3. Testing  
	a. How does CodeSearchNet Implement Testing?  
	b. Metrics for Test Accuracy: MRR  
	c. Metrics for Test Accuracy: nDCG  

4. Pre-trained model results

## 1. Neural Bag of Words Model
### 1a. Bag of Words Model
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


### 1b. Neural Bag of Words
The NBOW model takes an average of the word vectors in the input text and performs classification with a logistic regression layer. Essentially the NBOW model is a fully connected feed forward network with BOW input. [[2]](https://www.aclweb.org/anthology/W16-1626.pdf)


### 1c. CodeSearchNet Baseline Model: Neural Bag of Words
The Neural Bag of Words baseline model used by CodeSearchNet is interesting because it only uses Bag of Words to create tokens and then those tokens are fed into a learnable embedding to create the vector representation. 

Word embedding differs from bag of words. Bag of words suffers from high dimensionality and sparsity. The total number of unique words comprising the vocabulary can be huge, and given any particular document, its vector representation can be mostly zeros. This is not good for neural networks, so word embedding is a different technique that aims to reduce the dimensionality of a words representation. 

"In an embedding, words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space.

The position of a word within the vector space is learned from text and is based on the words that surround the word when it is used." [[9]](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)

The position of a word in the learned vector space is referred to as its embedding.

<div align="center"><img src="https://github.com/jawooson/AI_Project_3/blob/jason-dev/images/figure_3.png" width=65%/></div>

 Figure taken from [[1]](https://arxiv.org/pdf/1909.09436.pdf)


In the Neural Bag of Words Model, the code to be evaluated and the NLP query are made into tokens. This is the bag of words stage in the nBoW model. After tokens are made, they are fed into a word embedder (sequence encoder). This creates vector representations, of the code and the query, in a predefined vector space (ex R^4). Finally, the distance is measured between the code and the query, which creates the ranking from closest to furthest. The default distance formula used is the cosine distance.

Cosine distance is commonly used in tasks utilizing text data because Euclidean distance can be skewed based on document sizes. Cosine distance is a more appropriate measurement because the angle of vectors is measured, which makes it a more robust distance measurement. [[12]](https://www.machinelearningplus.com/nlp/cosine-similarity/)

## 2. Bidirectional RNNs model
### 2a. Basic RNN model
The idea behind RNNs is to make use of sequential information. Different from other neural network which assumes all inputs and outputs are independent, RNNs perform the same task for every element of a sequence, with the output being depended on the previous computations. It implements the same function *theta* across the sequence 1 :t. 

> ht = f(h(t-1), xt; *theta*) (From class slide)

For example, in NLP tasks, a RNN model treats each word of a sentence as a separate input occurring at time *t* and uses the activation value at *t-1* also, as an input in addition to the input at time *t*.

### 2b. Bidirectional RNN model

##### Why Bidirectional RNN?

In the RNNs model described above, the NN is forward, which means the architectures effects of occurrences at only the previous time stamps can be taken into account.
In the use of NLP, the RNNs only takes into account the effects of the word written only before the current word. 

  Example: 
 >1. ***He said, "Teddy** bears are on sale!"*
 >2. ***He said, "Teddy** Roosevelt was a great President!"* [\[15\]](https://www.coursera.org/lecture/nlp-sequence-models/bidirectional-rnn-fyXnn)
 
 In sentence 1 and 2, if we only look at the first three words ***He said, "Teddy***, we can't infer whether it's talking about a Teddy bear or the name of the president. So basic RNNs model doesn't work well regarding the language structure like this. 
 
 ![RNN model with example sentence](https://github.com/jawooson/AI_Project_3/blob/rzdev/images/RNN_w_eg.png)

##### What's Bidirectional RNN?

A bi-directional RNN consists of a forward and a backward recurrent neural network and final prediction is made combining the results of both the networks at any given time t. [\[14\]](https://towardsdatascience.com/natural-language-processing-from-basics-to-using-rnn-and-lstm-ef6779e4ae66)

Instead of running an RNN only in the forward mode starting from the first symbol, BRNN starts another one from the last symbol running from back to front by adding a hidden layer that passes information in a backward direction to more flexibly process such information.

i.e., in the previous ***He said, "Teddy*** example, BRNN will also look at words that appear after the first three words, and when it sees ***bears*** , it can infer that ***Teddy***  is refer to ***Teddy bears***.

![BRNN model with example sentence](https://github.com/jawooson/AI_Project_3/blob/rzdev/images/BRNN_w_eg.png)

### 2c. Gated Recurrent Unit

Another extension of basic RNNs used in the CodeSearchNet's baseline model is Gated Recurrent Unit (GRU). The difference between GRU and simple RNN block is that the GRU consists of an additional memory unit (commonly referred as an update gate or reset gate).

Besides the simple RNN unit with sigmoid function and a softmax for output, GRU has an additional unit with tanh as an activation function. The output from this unit is then combined with the activation input to update the value of the memory cell.[\[16\]](https://arxiv.org/abs/1412.3555)

Thus at each step value of both the hidden unit and the memory unit are updated. The value in the memory unit, plays a role in deciding the value of activation being passed on to the next unit.

  Example: 
 >*The **cat**, which already ate ...., **was** full.* [\[15\]](https://www.coursera.org/lecture/nlp-sequence-models/gated-recurrent-unit-gru-agZiL)

As it reads the above sentence from left to right, the GRU unit is going to have a new var called $C$, which stands for memory cell and it will provide a bit of memory to remember whether ***cat*** is singular or plural, so when it gets further into the sentence (when reaches the word ***was***) it can still work under consideration whether the subject of the sentence was singular or plural.

### 2d. CodeSearchNet Baseline Model: Bidirectional RNN model (with GRU)

CodeSearchNet uses BRNN as one of the sequence processing techniques architecture. It employs one encoder per input (natural or programming) language and trains them to map inputs into a single, joint vector space. 

>The objective is to map code and the corresponding language onto vectors that are near to each other, as we can then implement a search method by embedding the query and then returning the set of code snippets that are “near” in embedding space. [\[1\]](https://arxiv.org/pdf/1909.09436.pdf)

The BRNN architecture takes token sequences that are preprocessed according to their semantics (identifiers appearing in code tokens are split into subtokens and )and natural language tokens are split using byte-pair encoding (BPE) as input. Then output their contextualized token embeddings. 

Let's take look at  CodeSearchNet's baseline BRNN model's sequence encoder : **rnn_seq_encoder.py** and see what happens from input token sequences to contextualized token embeddings. 

 ![BRNN model encoder py code](https://github.com/jawooson/AI_Project_3/blob/rzdev/images/BRNN_encoder.png)

After the encoder return the BRNN's final state and token embeddings, a pooling will be implemented on the returned token embeddings. 

## 3. Self Attention & CNN Self Attention
### 3a. Introduction
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. It takes in *n* inputs, and returns *n* outputs. It allows the inputs to interact with each other (“self”) and find out who they should pay more attention to (“attention”). The outputs are aggregates of these interactions and attention scores, computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

CNN Self Attention is similar to attention model, with a convolutionary neural network as additional layers at the begining.

### 3b. Example
Say ”The baby panda couldn't reach the milk on the table because **it** is tiny. ”

What does “it” in this sentence refer to? Is it referring to the panda or to the milk? As intuitive as it sounds to human, it is a tricky question to the machine.

As the model processes each word in the sequence, self-attention differentiate itself from other machine learning models by allowing the model to link “it” with “panda”. It computes other positions in the input sequence for clues to build a better encoding for this word.

<img src="https://github.com/jawooson/AI_Project_3/blob/lldev/image/example2.png" height="400"/>

### 3c. Model explain
1. Preprocessing inputs

2. Initialize weights

3. Derive **key**, **query** and **value**

4. Compute attention scores for Input 1

5. Compute softmax

6. Multiply scores with **values**

7. Calculate **weighted** **mean** to get Output 1

8. Repeat steps 4–7 for Input 2 & Input 3

![image-20200508190210159](https://github.com/jawooson/AI_Project_3/blob/lldev/image/1.png)
Figure taken from [[17]]
![image-20200508145035167](https://github.com/jawooson/AI_Project_3/blob/lldev/image/2.png)
Figure taken from [[17]]
```python
encoder_hypers = {
  # CNN layers
  '1dcnn_position_encoding': 'none',
  '1dcnn_layer_list': [128, 128],
  '1dcnn_kernel_width': [8, 8], 
  '1dcnn_add_residual_connections': True,
  '1dcnn_activation': 'tanh',
  
  # Attention layers
  'self_attention_activation': 'gelu',
  'self_attention_hidden_size': 128,
  'self_attention_intermediate_size': 512,
  'self_attention_num_layers': 2,
  'self_attention_num_heads': 8,
  'self_attention_pool_mode': 'weighted_mean',
}
```
##### CNN layer

Activation function: tanh

##### Attention layer

Activation function: gelu (Gaussian Error Linear Unit activation function )

2 layers with 8 heads

128 hidden sizes

512 intermediate sizes



| tanh                                          | GELU                                        |
| --------------------------------------------- | ------------------------------------------------ |
| <img src="https://github.com/jawooson/AI_Project_3/blob/lldev/image/3.png" alt="3" width="400" /> | <img src="https://github.com/jawooson/AI_Project_3/blob/lldev/image/gelu.png" alt="3" width="400" /> |



**Encoder**

The encoder in the proposed Transformer model has multiple “encoder self attention” layers. Each layer is constructed as follows:

1. The input will be the word embeddings for the first layer. For subsequent layers, it will be the output of previous layer.
2. Inside each layer, first the multi-head self attention is computed using the inputs for the layer as keys, queries and values.
3. The output of #2 is sent to a feed-forward network layer. Here every position (i.e. every word representation) is fed through the same feed-forward that contains two linear transformations followed by a GeLU (input vector ->linear transformed hidden1->linear transformed hidden2 ->GeLU output).

```python
def make_self_attention_encoder:
		
          	# Step 1
            embbed_layer(tokens)
            
            # Step 2/3
            for every layer:
                compute self-attention layer based on inputs
                Add residual connections past the first layer
                GELU activate/ dropout
                
						# Return encoder with configured pool mode
```

Input: tokens, Output: self-attention encoder

**Decoder**

The decoder will also have multiple layers. Each layer is constructed as follows:

1. The input will be the word embeddings generated so far for the first layer. For subsequent layers, it will be the output of previous layer.
2. Inside each layer, first the multi-head self attention is computed using the inputs for the layer as keys, queries and values (i.e. generated decoder outputs so far, padded for rest of positions).
3. The output of #2 is sent to a “multi-head-encoder-decoder-attention” layer. Here yet another attention is computed using #2 outputs as queries and encoder outputs as keys and values.
4. The output of #3 is sent to a position wise feed-forward network layer like in encoder.

## 4. Testing
### 4a. How does CodeSearchNet Implement Testing?  
The test set consists of 99 queries. For each query, we are given 1000 code snippets. Of the 1000 code snippets, only one is relevant and 999 are distractors, so the evaluation task is to rank them.  [[1]](https://arxiv.org/pdf/1909.09436.pdf) 

### 4b. Metrics for Test Accuracy: MRR  
Mean Reciprocal Rank is very simple. It measures where the first relevant term is. So, given CodeSearchNet, it measures where the one relevant code snippet is positioned relative to the other 999. Given its absolute rank, find the reciprocal. For example, if for one query the model positions in the 6th slot, it is computer as 1/6. Given that we have 99 different queries, MRR just takes the mean of all these reciprocal ranks. Again, for example if there are 3 queries with the first reciprocal rank given above and the other two are 1/8 and 1, the MRR is 1/3*(1/6 + 1/8 + 1). 

This metric used for accuracy is much better than traditional accuracy score because it deals with ranked data. Given a particular query and a test set, we want to find the most relevant code snippet compared. It is clear why rank is essential in testing because order is essential.
[[13]](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)


### 4c. Metrics for Test Accuracy: nDCG
* Normalized Discounted Cumulative Gain is used for the W&B rankings because it takes into account different users running different models. I won't discuss the derivation of nDCG too heavily, but it is good at capturing the ranking of relevant documents, as well as varying number of test documents. 

<div align="center"><img src="https://github.com/jawooson/AI_Project_3/blob/jason-dev/images/ndcg_diagram.png" width=65%/></div>

 Figure taken from [[13]](https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1).

## 5. Pre-trained Model Results

| (bs=1,000)              | EXPV  | 1DCNN | BOW   | Self Attention | CNN Self Attention | BRNN  |
| ----------------------- | ----- | ----- | ----- | -------------- | ------------------ | ----- |
| Test-python MRR         | 0.572 | 0.579 | 0.586 | **0.692**          | 0.632              | 0.638 |
| FuncNameTest-python MRR | 0.469 | 0.577 | 0.48  | **0.68**           | 0.595              | 0.644 |
| Validation-python MRR   | 0.542 | 0.529 | 0.559 | **0.643**          | 0.583              | 0.588 |

![img](https://github.com/jawooson/AI_Project_3/blob/lldev/image/comp.png)

## Bibliography

## 6. Conclusion
Overall all the models in the baseline perform pretty similarly with MRR score ranging between 0.47-~0.69. Even though there's definitely room for improvement, running a model can take multiple hours even overnight. Self Attention has the overall highest MRR score in all three categories (Test/ FuncNameTest/ Validation). BOW has the lowest MRR score out of three models we discussed across all three categories.

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
[14] [Natural Language Processing: From Basics to using RNN and LSTM](https://towardsdatascience.com/natural-language-processing-from-basics-to-using-rnn-and-lstm-ef6779e4ae66)  
[15]  [Bidirectional RNN Online Course Taught by Andrew Ng](https://www.coursera.org/lecture/nlp-sequence-models/bidirectional-rnn-fyXnn)  
[16] [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555)  
[17] [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)  
[18] [Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)  
[19] [Attn: Illustrated Attention](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)  
[20] [Illustrated transformer](https://jalammar.github.io/illustrated-transformer/)  
[21] [On the relationship between self-attention and convolutional layers](https://openreview.net/pdf?id=HJlnC1rKPB)  

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0OTA0MTA2MTIsNTQ1OTY4MzMzLDE4ND
Q3OTc1MywxMjQ3NzUxMTEyLDU4MzU0MDQ1MywtMTE0Njk0ODQ3
MywtMTc2ODY1NjQyOCwxMTA2NTY3NTU5LDExMzM0MTg4NzYsOT
YwODQ4OTY4LDE3ODY0MDk5OTksMTExNzY5MzQ4NiwxNDQxNTY2
NTA5LDIwNzI3NzM1MywtNTg2NTMwNjcyLC0xMTc2MjQ4MjM1LD
EzOTM4OTc4NiwxNTUyMTM2NjksMTc5MDcxMDI3Miw5NjA1Nzgx
NDZdfQ==
-->
