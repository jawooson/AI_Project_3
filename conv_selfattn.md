
## Self Attention & CNN self-attention
## 1. Introduction

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. It takes in *n* inputs, and returns *n* outputs. It allows the inputs to interact with each other (“self”) and find out who they should pay more attention to (“attention”). The outputs are aggregates of these interactions and attention scores, computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

CNN Self Attention is similar to attention model, with a convolutionary neural network as additional layers at the begining.



## 2. Example

Say ”The animal didn't cross the street because **it** was too tired”

What does “it” in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.

When the model is processing the word “it”, self-attention allows it to associate “it” with “animal”.

As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

<img src="image/example.png" alt="3" width="400" />


## 3. Model explain
1. Prepare inputs

2. Initialise weights

3. Derive **key**, **query** and **value**

4. Calculate attention scores for Input 1

5. Calculate softmax

6. Multiply scores with **values**

7. Calculate **weighted** **mean** to get Output 1

8. Repeat steps 4–7 for Input 2 & Input 3

   ![image-20200508190210159](image/1.png)

   

   ![image-20200508145035167](image/2.png)

   

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

| tanh                                          | Self Attn                                        |
| --------------------------------------------- | ------------------------------------------------ |
| <img src="image/3.png" alt="3" width="400" /> | <img src="image/gelu.png" alt="3" width="400" /> |



**Encoder**

The encoder in the proposed Transformer model has multiple “encoder self attention” layers. Each layer is constructed as follows:

1. The input will be the word embeddings for the first layer. For subsequent layers, it will be the output of previous layer.
2. Inside each layer, first the multi-head self attention is computed using the inputs for the layer as keys, queries and values.
3. The output of #2 is sent to a feed-forward network layer. Here every position (i.e. every word representation) is fed through the same feed-forward that contains two linear transformations followed by a GeLU (input vector ->linear transformed hidden1->linear transformed hidden2 ->GeLU output).

```python
def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope("self_attention_encoder"):
            self._make_placeholders()
						
          	# Step 1
            seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])

            activation_fun = get_activation(self.get_hyper('1dcnn_activation'))
            current_embeddings = seq_tokens_embeddings
            num_filters_and_width = zip(
              												self.get_hyper('1dcnn_layer_list'),
              												self.get_hyper('1dcnn_kernel_width'))
            
            # Step 2
            for (layer_idx,(num_filters, kernel_width)) in enumerate(num_filters_and_width):
                next_embeddings = tf.layers.conv1d(
                    inputs=current_embeddings,
                    filters=num_filters,
                    kernel_size=kernel_width,
                    padding="same")

                # Add residual connections past the first layer.
                if self.get_hyper('1dcnn_add_residual_connections') and layer_idx > 0:
                    next_embeddings += current_embeddings

                current_embeddings = activation_fun(next_embeddings)

                current_embeddings = tf.nn.dropout(
                  											current_embeddings,
                                       	keep_prob=self.placeholders['dropout_keep_rate'])
						# Step 3
            config = BertConfig(
              vocab_size=self.get_hyper('token_vocab_size'),
              hidden_size=self.get_hyper('self_attention_hidden_size'),
              num_hidden_layers=self.get_hyper('self_attention_num_layers'),
              num_attention_heads=self.get_hyper('self_attention_num_heads'),
              intermediate_size=self.get_hyper('self_attention_intermediate_size'))

            model = BertModel(config=config,
            	is_training=is_train,
              input_ids=self.placeholders['tokens'],
              input_mask=self.placeholders['tokens_mask'],
              use_one_hot_embeddings=False,
              embedded_input=current_embeddings)

            output_pool_mode = self.get_hyper('self_attention_pool_mode').lower()
            if output_pool_mode == 'bert':
                return model.get_pooled_output()
            else:
                seq_token_embeddings = model.get_sequence_output()
                seq_token_masks = self.placeholders['tokens_mask']
                seq_token_lengths = tf.reduce_sum(seq_token_masks, axis=1)
                return pool_sequence_embedding(
                  			output_pool_mode,
                       	sequence_token_embeddings=seq_token_embeddings,
                  			sequence_lengths=seq_token_lengths,
                  			sequence_token_masks=seq_token_masks)
```



**Decoder**

The decoder will also have multiple layers. Each layer is constructed as follows:

1. The input will be the word embeddings generated so far for the first layer. For subsequent layers, it will be the output of previous layer.
2. Inside each layer, first the multi-head self attention is computed using the inputs for the layer as keys, queries and values (i.e. generated decoder outputs so far, padded for rest of positions).
3. The output of #2 is sent to a “multi-head-encoder-decoder-attention” layer. Here yet another attention is computed using #2 outputs as queries and encoder outputs as keys and values.
4. The output of #3 is sent to a position wise feed-forward network layer like in encoder.

## 4. Metrics  

| (bs=1,000)              | Self Attn | CNN Self Attn |
| ----------------------- | --------- | ------------- |
| Test-python MRR         | 0.692     | 0.632         |
| FuncNameTest-python MRR | 0.680     | 0.595         |
| Validation-python MRR   | 0.643     | 0.583         |


## Bibliography

<a href="https://arxiv.org/pdf/1706.03762.pdf">Attention is all you need</a>

<a href="https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a">Illustrated: Self-Attention</a>

<a href="https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3">Attn: Illustrated Attention</a>

<a href="https://jalammar.github.io/illustrated-transformer/">Illustrated transformer</a>

<a href="https://openreview.net/pdf?id=HJlnC1rKPB">On the relationship between self-attention and convolutional layers</a>

