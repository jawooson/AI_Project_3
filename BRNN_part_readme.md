### Bidirectional RNN model

#### - Basic RNN model

 The idea behind RNNs is to make use of sequential information. Different from other neural network which assumes all inputs and outputs are independent, RNNs perform the same task for every element of a sequence, with the output being depended on the previous computations.
It implements the same function (parametrized by $\theta$) across the sequence $1:\tau$ 

> $$h_t = f(h_{(t-1)},x_t;\theta)$$ [\[1\]](https://pantelis.github.io/cs-gy-6613-spring-2020/docs/lectures/rnn/introduction/)

For example, in NLP tasks, a RNN model treats each word of a sentence as a separate input occurring at time $t$ and uses the activation value at $t-1$ also, as an input in addition to the input at time $t$.
 
 #### - Bidirectional RNN model
 ##### Why Bidirectional RNN?
In the RNNs model described above, the NN is forward, which means the architectures effects of occurrences at only the previous time stamps can be taken into account.

In the use of NLP, the RNNs only takes into account the effects of the word written only before the current word. 

  Example: 
 >1. ***He said, "Teddy** bears are on sale!"*
 >2. ***He said, "Teddy** Roosevelt was a great President!"* [\[2\]](https://www.coursera.org/lecture/nlp-sequence-models/bidirectional-rnn-fyXnn)
 
 In sentence 1 and 2, if we only look at the first three words ***He said, "Teddy***, we can't infer whether it's talking about a Teddy bear or the name of the president. So basic RNNs model doesn't work well regarding the language structure like this. 
 
 ![RNN model with example sentence](https://github.com/jawooson/AI_Project_3/blob/rzdev/images/RNN_w_eg.png)

##### What's Bidirectional RNN?

A bi-directional RNN consists of a forward and a backward recurrent neural network and final prediction is made combining the results of both the networks at any given time t. [\[3\]](https://towardsdatascience.com/natural-language-processing-from-basics-to-using-rnn-and-lstm-ef6779e4ae66)

Instead of running an RNN only in the forward mode starting from the first symbol, BRNN starts another one from the last symbol running from back to front by adding a hidden layer that passes information in a backward direction to more flexibly process such information.

i.e., in the previous ***He said, "Teddy*** example, BRNN will also look at words that appear after the first three words, and when it sees ***bears*** , it can infer that ***Teddy***  is refer to ***Teddy bears***.

![BRNN model with example sentence](https://github.com/jawooson/AI_Project_3/blob/rzdev/images/BRNN_w_eg.png)
 
 #### - Gated Recurrent Unit

Another extension of basic RNNs used in the CodeSearchNet's baseline model is Gated Recurrent Unit (GRU). The difference between GRU and simple RNN block is that the GRU consists of an additional memory unit (commonly referred as an update gate or reset gate).

Besides the simple RNN unit with sigmoid function and a softmax for output, GRU has an additional unit with tanh as an activation function. The output from this unit is then combined with the activation input to update the value of the memory cell.[\[6\]](https://arxiv.org/abs/1412.3555)

Thus at each step value of both the hidden unit and the memory unit are updated. The value in the memory unit, plays a role in deciding the value of activation being passed on to the next unit.

  Example: 
 >*The **cat**, which already ate ...., **was** full.* [\[5\]](https://www.coursera.org/lecture/nlp-sequence-models/gated-recurrent-unit-gru-agZiL)

As it reads the above sentence from left to right, the GRU unit is going to have a new var called $C$, which stands for memory cell and it will provide a bit of memory to remember whether ***cat*** is singular or plural, so when it gets further into the sentence (when reaches the word ***was***) it can still work under consideration whether the subject of the sentence was singular or plural.

#### - CodeSearchNet Baseline Model: Bidirectional RNN model (with GRU)

CodeSearchNet uses BRNN as one of the sequence processing techniques architecture. It employs one encoder per input (natural or programming) language and trains them to map inputs into a single, joint vector space. 

>The objective is to map code and the corresponding language onto vectors that are near to each other, as we can then implement a search method by embedding the query and then returning the set of code snippets that are “near” in embedding space. [\[7\]](https://arxiv.org/pdf/1909.09436.pdf)

The BRNN architecture takes token sequences that are preprocessed according to their semantics (identifiers appearing in code tokens are split into subtokens and )and natural language tokens are split using byte-pair encoding (BPE) as input. Then output their contextualized token embeddings. 

Let's take look at  CodeSearchNet's baseline BRNN model's sequence encoder : **rnn_seq_encoder.py** and see what happens from input token sequences to contextualized token embeddings. 

 ![BRNN model encoder py code](https://github.com/jawooson/AI_Project_3/blob/rzdev/images/BRNN_encoder.png)

After the encoder return the BRNN's final state and token embeddings, a pooling will be implemented on the returned token embeddings. 

