1. Can you identify some inefficiencies in the tiny LLaMA implementation?

* Each encoder block has one `RoPEMultiHeadedAttention` which stores a copy of the RoPE frequencies matrix. This is redundant. It is possible to create a single copy and pass it to the attention call every time.
* The causal mask in the `attention` function in `RoPEMultiHeadedAttention` is created every time when it is being called. Instead, it can be created as a static mask and then passed to the function during calling.
