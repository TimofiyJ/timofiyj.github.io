---
layout: post
title: "Attention is all you need paper summary"
author: "Tymofii Nasobko"
categories: journal
tags: [papers,review, summary]
image: attention-is-all-you-need.png
---

# Attention is all you need

## Introduction

**Attention is all you need** is the 2017 paper that created new architecture called "Transformers", essentially making breakthrough in AI and Deep Learning with GPT that is based on that architecture. The main innovation of this paper is that authors do not use RNNs or Convolutions and purely focus on Self-Attention mechanism that allowed massive parallelization.

## Background
Authors bring up **RNN**, **CNN**, **LSTM**, **GRNN** and **Encoder-Decoder** architectures and critisize their sequential nature which disables parallelization  as memory constraints limit batching across examples. Reader should be familiar with such architectures to understand the main advantages of the proposed approach.
Self-attention - an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence

## Architecture
 The Transformer architecture is based on **Encoder-Decoder** model:
 > where encoder maps an input sequence of symbol representations _(x1, ..., xn)_ to a sequence of continuous representations _$z$ = (z1, ..., zn)_. Given $z$, the decoder then generates an output sequence _(y1, ..., ym)_ of symbols one element at a time. At each step the model is auto-regressive consuming the previously generated symbols as additional input when generating the next.

 ![Transformer architecture](assets/img/attention-is-all-you-need.png)

>The Transformer follows this overall architecture using stacked self-attention and point-wise, fully
connected layers for both the encoder and decoder

### Encoder
The encoder consists of 6 stacked identical layers, each layer has 2 sub-layers: the first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. Authors used residual connection for each of the two sub-layers, followed by normalization layer (Add & Norm step). That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension 512.

---
<details>
<summary><b>Notes</b></summary>
While Self-Attention and FFN are intuitively understandable, Add & Norm step is pretty confusing, so here is additional explanation:

1. The Mathematical Formula
   
   The paper defines the output of a block as:
   $$
   \text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
   $$
   Where:
   - $x$: The input that entered the sub-layer (e.g., the raw word embeddings entering the Attention layer)
   - $\text{Sublayer}(x)$: The "transformation" (either the Multi-Head Attention or the FFN)
   - $x + \text{Sublayer}(x)$: The Residual Connection (also called a "skip connection")
2. The Residual Connection ("Add") 

   Instead of the layer trying to learn the entire transformation from scratch, it only learns the difference (the residual) that needs to be added to the input. <br>
   - The Problem: <br>
   In very deep models (like the 6-stack Transformer), gradients can "vanish" or get lost as they flow back through many layers during training. <br>
   - The Solution: <br>
   The $+ x$ creates a "highway" for the gradient. Even if the weights in the Sublayer are doing a poor job early in training, the original information $x$ can still pass through unimpeded. This makes the network much easier to train and allows it to be much deeper.
3. Layer Normalization ("Norm")
   
   After adding the original input back, the model applies Layer Normalization. <br>
   - What it does: <br>
   It re-scales the values across the features of a single sequence element (a single word/token) so they have a mean of 0 and a variance of 1.
   - Why it's there: <br>
    It prevents the values (activations) from exploding to massive numbers or shrinking to near-zero. It keeps the "signals" inside the network stable across all layers.
4. Putting it all together
   
   In a single Transformer Encoder block, this happens twice: <br>
   - First Sub-layer: <br>
    $\text{LayerNorm}(x + \text{MultiHeadAttention}(x))$
   - Second Sub-layer: <br>
    $\text{LayerNorm}(x + \text{FeedForward}(x))$
</details>

---

### Decoder
The decoder also consists of 6 layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Risidual connection is also applied between the sub-layers, followed by normalization. "Masking" and "Offset" trick is added: before calculating the softmax in the attention mechanism, the attention scores for all "future" positions are set to $-\infty$ (negative infinity). Offset by one position refers to how the training data is fed into the decoder. This is often called Teacher Forcing.

If the target sentence is:<br>
[START] I love AI [END] <br>
Input to Decoder: [START] I love AI <br>
Expected Output: I love AI [END] <br>

By shifting (offsetting) the output, the model learns that when it sees [START], it must predict I. When it sees [START] I, it must predict love.

Because of the masking + the offset, the prediction for the next word is based only on the words that came before it.

### Attention
As the paper says:
> An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

Before attention happens, each token is an embedding vector (let's call it $x$). To get $Q, K, \text{ and } V$, we multiply that $x$ by three different weight matrices ($W^Q, W^K, W^V$). <br>
At the end, the Output is also a vector. It’s the result of taking the Values and mixing them together based on the attention scores. If the input was "bank," the output vector is "bank" but with the "river" context mixed into its numbers.

The dimensions of $Q, K, V$ are:
- $d_{model}$ = 512: This is the size of the original embedding
- $d_k$ = 64: Query and Key dimension
- $d_v$ = 64: Value dimension
  
Since Multi-Head Attention has 8 heads we have 64 instead of 512 - they split the 512-dimensional space into 8 smaller "sub-spaces.". Each head focuses on a different type of relationship (one head might look for grammar, another for phrasal verbs, another for punctuation) using its own 64-dimensional vectors. At the end, they glue the 8 results back together to get 512 again.

---

<details>
<summary><b>Notes</b></summary>
Every word in a sentence acts as a Query looking at all other words (the Keys) to see how relevant they are.

**Example:** If the sentence is "The bank of the river," the word "bank" is the Query. It looks at the Key for "river" and realizes there is a high "compatibility."

The "Output" is the new, context-aware version of the word.
Instead of just taking the most relevant word, the model takes a little bit of information from every word in the sentence. However, it takes more from the important words and less from the unimportant ones.

The model multiplies the Query vector by the Key vector ($Q \cdot K$).If the vectors point in the same direction (they are similar), the result is a large number.If they are perpendicular (unrelated), the result is small.These raw scores are passed through a Softmax function to turn them into probabilities (weights) that add up to 100%.

You may ask: Can a combination of keys be used? (e.g., phrasal verbs) <br>
Yes, but indirectly. A single Query doesn't just match one Key; it gets a "match score" for every Key in the sentence.
If you have a phrasal verb like "look up," the Query for "look" will likely have high attention scores for the Key of "up."
The model then takes a weighted sum of their Values.
Because the Transformer has multiple layers, by the time you get to Layer 2 or 3, the vector for "look" already contains information about "up." The "combination" happens as the data flows deeper into the stack.
</details>

---

#### Scaled Dot-Product Attention
Input: queries and keys of dimension ${d_k}$, and values of dimension ${d_v}$. <br>
Dot product of the query with all keys is computed and each divided by ${\sqrt d_k}$, applied softmax function to obtain the weights on the
values.

Attention function is computed on the a set of queries simultaneously, packed into a matrix $Q$. The keys and values are also packed together into matrices $K$ and $V$. <br>
Matrix of outputs is computed as:
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
> The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
of ${\sqrt \frac{1}{d_k}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

> While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of ${d_k}$. We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by ${\sqrt \frac{1}{d_k}}$.

#### Multi-Head Attention
> Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values


> Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this.

$$
MultiHead(Q,K,V) = Concat(head_1 ... head_h)W^O \\
\text{where } head_i = Attention(QW_i^Q, KW _i^K, VW_i^V)
$$

> The Transformer uses multi-head attention in three different ways:
>  - In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence.
> - The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
> - Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.

___
<details>
<summary><b>Notes</b></summary>
In the Encoder, the goal is to understand the input sentence perfectly by looking at the context.
<ul>
  <li>Where do Q, K, V come from? All three come from the same place: the output of the previous encoder layer.</li>
  <li>What it does: Every word in the input sentence looks at every other word in that same sentence.</li>
  <li>Example: If the input is "The bank of the river," the word "bank" ($Q$) looks at "river" ($K$) to realize it means land, not a building.</li>
</ul>

In the Decoder, the goal is to generate the next word of the output, but the model is only allowed to look at words it has already generated.
<ul>
  <li>Where do $Q, K, V$ come from? All three come from the decoder's own previous layers.</li>
  <li>The Constraint (Masking): This uses a mask ($-\infty$).</li>
  <li>What it does: The word at position i can look at positions $1$ through i, but is physically blocked from seeing i+1 (the future). This preserves the "auto-regressive" property—meaning the model predicts the future based only on the past.</li>
</ul>
Encoder-Decoder Attention: It connects the "meaning" found by the Encoder to the "generation" happening in the Decoder.
<ul>
  <li>The Query (Q): Comes from the Decoder. It represents the word the model is currently trying to translate/generate.</li>
  <li>The Keys (K) and Values (V): Come from the Encoder's final output. They represent the "memory" of the original input sentence.</li>
  <li>What it does: The Decoder says: "I am currently writing the third word of the French translation (Q). I need to look back at the original English sentence (K, V) to see which English words I should be focusing on right now."</li>
</ul>
</details>

---
#### Position-wise Feed-Forward Networks
> In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
connected feed-forward network, which is applied to each position separately and identically. This
consists of two linear transformations with a ReLU activation in between.
$$
FNN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$
> While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is $d_{model}$ = 512, and the inner-layer has dimensionality $d_{ff}$ = 2048.

In the Transformer architecture, the Position-wise Feed-Forward Network (FFN) acts as a "local processor" for each token. While the Attention layers are responsible for moving information between tokens (context gathering), the FFN is responsible for processing that information (feature extraction and refinement).

___
<details>
<summary><b>Notes</b></summary>
Why FFN is added?
<ul>
  <li>Adding Non-Linearity: The Self-Attention mechanism is essentially a series of linear operations (weighted averages). If you only had Attention layers, the entire Transformer would just be one giant linear transformation, which is mathematically limited in the complexity of patterns it can learn.</li>
  <li>Feature Expansion: It takes the 512-dimensional input and projects it into a 2048-dimensional space, applies the activation function in this massive space, projects the 2048 dimensions back down to 512 to match the residual connection. This provides the model with more "computational room" to combine different features gathered by the attention heads before condensing them back into a single vector.</li>
  <li>Acting as "Key-Value Memory": It is suggested that FFN layers actually store the "knowledge" of the model. The first linear layer acts as a set of pattern detectors, The second linear layer acts as the responses to those patterns</li>
</ul>
</details>

___

#### Embeddings and Softmax
Learned embeddings are used to transform input and output tokens into vectors of dimension $d_{model}$. Learned linear transformation and softmax function are used to convert the decoder output to predicted next-token probabilities. The same weight matrix is shared between two embedding layers and the pre-softmax
linear transformation. In the embedding layers weights are multiplied by $\sqrt{d_{model}}$

#### Positional encoding
> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed. 
> 
> In this work, we use sine and cosine functions of different frequencies:
$$
P E(_{pos, 2i}) = sin(pos/1000^{2i/d_{model}}) \\
P E(_{pos, 2i+1}) = cos(pos/1000^{2i/d_{model}})
$$
> Where $pos$ is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $P E_{pos+k}$ can be represented as a linear function of P $E_{pos}$.

---
<details>
<summary><b>Notes</b></summary>
Since the Transformer processes all tokens simultaneously (in parallel), it is naturally "position-invariant"—meaning it treats the sentence "The dog bit the man" exactly the same as "The man bit the dog."
The Positional Encoding is the solution to this "bag-of-words" problem.

The goal is to add information to each token's embedding that indicates its specific position in the sequence.

Instead of concatenating a number (which would change the dimension of the vector) or using a simple counter (which could grow too large for long sequences), the authors add a "positional signal" vector directly to the embedding vector.
For this to work, the Positional Encoding must satisfy three main criteria:
It must be unique for every position; The distance between any two positions should be consistent across sentences of different lengths; It should allow the model to generalize to sequence lengths longer than those seen during training.

The authors hypothesized that using sine and cosine would allow the model to easily learn to attend by relative positions. Because of the trigonometric identity
$$
\sin(a+b) = \sin(a)\cos(b) + \cos(a)\sin(b)
$$
The encoding for position $pos + k$ can be represented as a linear function of the encoding at position $pos$. This makes it mathematically "visible" to the model that Word A is $k$ steps away from Word B.

Why not just "Learn" the positions?<br>
You can learn positional embeddings (like a lookup table), however, the authors of the original paper chose the sinusoidal version because:<br>
Efficiency: It doesn't add any trainable parameters to the model.<br>
Generalization: They suspected the sinusoidal version would allow the model to handle much longer sequences than a learned version, which is limited by the size of the lookup table defined at the start.
</details>

---

## Useful resources
[Illustrations for Transformer](https://jalammar.github.io/illustrated-transformer/)