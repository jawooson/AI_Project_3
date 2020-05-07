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

i.e., in the previous ***He said, "Teddy*** example, BRNN will also look at words that appear after the first three words, and when it sees ***bears*** , it can infer that ***Teddy***  is refer to ***Teddy bears***.

![BRNN model with example sentence](https://github.com/jawooson/AI_Project_3/blob/rzdev/images/BRNN_w_eg.png)
 
