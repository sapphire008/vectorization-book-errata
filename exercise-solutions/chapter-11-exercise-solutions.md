1. What are the differences between encoder and decoder architecture in the transformer model?
* The encoder uses bi-directional attention. This means, that past tokens can be used to predict future tokens, while future tokens can also be used to predict past tokens.
* The decoder uses masked, or causal attention. Past tokens can be used to predict future tokens, but not vice versa. In Chapter 5 when we were discussing the scaled-dot-product attention mechanism, we have illustrated the two types of attention mechanism.
* Specific to the original Transformer paper, the decoder block also receives inputs from the hidden state of the encoder layer with an additional attention layer. Therefore, the encoder layer only has one attention layer, while the decoder layer has 2 attention layers. However, in more recent architectures such as Generative Pretrained Transformer (GPT), decoder architecture specifically refers to a transformer block that has only one causal attention layer.

2. What is Cloze task? How is it being used to pre-train the language model?

Cloze task is the fill-in-the-blank task, where the model learns to predict the missing word. Researchers use a large body of text corpus to generate self-supervised, self-labeled training data by blanking out words randomly from a sentence.

3. What are the differences between Pre-LN and Post-LN architecture?

* Pre-LN applies layer normalization immeditately to the input. Post-LN applies layer normalization after initial attention.
* Pre-LN requires a final layer normalization after a series of transformer blocks, while post-LN does not.
* Pre-LN architecture does not require a learning rate schedule for stable training, though it can still be benefitted from one.
* Pre-LN converges faster than Post-LN in certain tasks.
* Certain domains such as text-to-speech benefits more from Post-LN architecture than Pre-LN architecture.


4. Compare and contrast batch normalization and layer normalization.

Batch noralization applies normalization for each feture across different batches, while layer normalization applies normalization within the batch, across the sequence x feature or only the feature dimension. Both techniques can encourage training stability by regularizing the layer inputs.


5. How does RMSNorm differ from LayerNorm?

RMSNorm is a variant of LayerNorm. Instead of standarizing the inputs by shifting to zero mean and scale to unit variance, RMSNorm does not shift the mean and only scale based on the root-mean-square of the input.

6. What are the two formulations of the sinusoidal positional encoding?

Sinusoidal positional encodings incorporates absolute positional information about the tokens in the input sequence. The original transformer model interleaves the cosine and sine terms along the embedding dimension, while tensor2tensor model from Google stacks them together. In practice, these two formulations are equivalent.

7. How does the idea of sinusoidal positional encoding differ from that of rotary positional encoding?

The original transformer paper uses sinusoidal absolute position encoding added on top of the embedding sequence such that each embedding contains information about the absolute position of the sequence. In contrast, the idea of rotary positional encoding attempts to rotate the embedding vector to an angle based on the position of the token embedding. It is derived from a form of relative positional embedding. Recall that relative positional encoding is adding the positional encoding term on the dot product between query and key. The idea behind rotatry position encoding is that, if we rotate the query and key vector at the same position by the same amount of angle, then their dot product does not change, since dot product is defined as the product between the magnitude of the two vectors as well as the relative angle of the two vetors (cos $\theta$). 

8. Name several activation functions used in the feedforward layer of the transformer model.

Example activation functions that were previously explored are ReLU, ELU, GELU, and Swish. These are non-learnable activation functions. The gated linear unit (GLU) is a learnable activation function that uses a combination of swish activated and linear gates (i.e. two projections of the same inputs multiplied together).

9. What are topics of discussion with respect to AI safety and alignment?

Topics of heated debate in response to recent advances of AI research include but are not limited to: demographic biases, content toxcity, truthfulness and misinformation, ethnical and societal considerations, risk mitigation, policy and regulation. Each AI company dedicates large amount of research effort to address these issues to ensure better AI alignment.

10. Can you identify some inefficiencies in the tiny LLaMA implementation?

* Each encoder block has one `RoPEMultiHeadedAttention` which stores a copy of the RoPE frequencies matrix. This is redundant. It is possible to create a single copy and pass it to the attention call every time.
* The causal mask in the `attention` function in `RoPEMultiHeadedAttention` is created every time when it is being called. Instead, it can be created as a static mask and then passed to the function during calling.

