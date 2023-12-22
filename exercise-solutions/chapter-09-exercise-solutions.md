1. Use the DataFrame from Section 9.1 to answer the following question: How many total hours of viewership does each movie have? Hint: you might want to use `groupby` and `sum` to answer this question.


```python
import pandas as pd

df = pd.DataFrame(
    {
        "user": [1, 3, 2, 1, 2, 3, 1],
        "movie": [3, 2, 3, 2, 1, 3, 1],
        "viewing hours": [0.6, 1.3, 0.6, 0.7, 0.1, 0.5, 0.9],
    }
)

df.groupby(by=["movie"]).agg({"viewing hours": "sum"})

#        viewing hours
# movie               
# 1                1.0
# 2                2.0
# 3                1.7

```


2. What is the relationship between broadcasting and groupby operations?

Broadcasting can be seen as a groupby.apply pattern operating on matrices directly, where elements along the broadcasted dimension are indivdual groups.

3. Discuss on the disadvantages of `sliding_window_tf` in contrast to PyTorch and NumPy's sliding window operations (consider the concepts of `view` vs. `copy`).

`sliding_window_tf` always return a copy which can occupy large amount of memory.

4. Using the idea of bucketizaion and quantization, represent a set of float numbers between `[-1, 1]` (e.g. `[0.62290916, -0.37006572,  0.49293062, -0.87785846,  0.62497933]`) using 8-bit integers (i.e. between -128 and 127, total 256 numbers). What is the average error of this representation?

```python
import numpy as np
x = np.random.rand(10000) * 2 -1 # random between -1 and 1
buckets = np.linspace(-1, 1, 256)
x_digitized = np.digitize(x, buckets) - 129 # between -128 to 127
# Recover the numbers based on the buckets
x_recover = np.take(buckets, x_digitized + 129)
err = (x_recover - x) / x * 100
print(err.mean()) # -1.49 %
```

With a simulation of 10,000 random numbers, we saw an error rate of -1.5% of the digitized values.

5. Compare and contrast the three sets of segment-wise operations in Tensorflow.

`tf.math.segment_*` has the simplest assumptions: the grouping indices are sorted ascendingly, and there is no indices skipped; indices are consecutive. This assumption is then loosened in `tf.math.unsorted_segment_*` operations, where the grouping indices do not have to be sorted. The assumption is then further loosened in `tf.sparse.segment_*` operations, where it now operates on sparse tensors, and the grouping indices can skip over values.

6. Try to apply the `np_mean_reduceat` on the following data:

    ```python
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    indices = np.array([0, 3, 8])
    ```
    
    What is the result? Is this expected? Try to verify it using a simple Python for-loop.

```python
import numpy as np
# TODO import np_mean_reduceat
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
indices = np.array([0, 3, 8])

out = np_mean_reduceat(data, indices)
#  array([2., 6., 9.])

# Verify with for loop:
start_end = np.concatenate([indices, [len(data)]])
res = []
for ii in range(len(indices)):
    start = start_end[ii]
    end = start_end[ii+1]
    res.append(data[start:end].mean())

# [2.0, 6.0, 9.0]
```

7. Try to implement the logic of `tf_sparse_segment_prod` using PyTorch with `torch.scatter_add`.

```python
import torch

def torch_sparse_segment_prod(data, segment_ids, num_segments=None):
    if num_segments is None:
       num_segments = max(segment_ids) + 1
    # take log values
    data_out = torch.as_tensor(data).to(torch.float32)
    data_out = torch.log(data_out)
    segment_ids = segment_ids.to(torch.int64)
    # Segment sum of log
    out = torch.zeros(num_segments)
    out = out.scatter_reduce_(
        0, segment_ids, data_out, reduce="sum"
    )
    # Raise back to exponents
    out = torch.exp(out)
    return out.to(data.dtype)

data = torch.tensor([1, 2, 3, 2, 3, 6, 3, 4])
segment_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
out = torch_sparse_segment_prod(data, segment_ids)
```

8. Is there an alternative way to implement the function `torch_scatter_reduce_sqrt_n`, using `torch.segment_reduce`?

```python
import torch

def torch_segment_reduce_sqrt_n(arr, offsets):
    # Initialize accumulators
    arr = arr.to(torch.float32)
    counter = torch.ones_like(arr, dtype=torch.float32)
    # Compute groupwise sum
    gsum = torch.segment_reduce(arr, "sum", offsets=offsets)
    # Compute groupwise count
    gcount = torch.segment_reduce(counter, "sum", offsets=offsets)
    # Compute sum / sqrt_n
    return gsum / torch.sqrt(gcount)

arr = torch.tensor([1, 2, 5, 4, 7, 2, 3, 0, 1])
offsets = torch.tensor([0, 4, 5, 9])
out = torch_segment_reduce_sqrt_n(arr, offsets)
# tensor([6., 7., 3.])
```

9. Consider the relationships between sparse tensors, jagged tensors, and the segment-wise operations. What are the advantages and disadvantages of each?

Segment-wise operations, in its raw form, can compute over a single dimension of a dense tensor. It is efficient if the data inputs are already flattened along the dimension to be segment-summed. Otherwise, complicated transformation is needed preprocess and postprocess the shape of the data. Both Sparse Tensor and Jagged Tensor operations are efficient if the data is already in a sparse or jagged form. However, converting raw data into sparse tensor can be expensive.

10. Various types of positional encoding / embedding schemes have been proposed since the original Transformer paper. One way to encode positions is to use learned embeddings (where the positions are treated as categorical / ordinal features). Try to implement an embedding based positional encodings in PyTorch. Replace the sinusoidal encoding with the learnable embeddings in the ViT model and try to retrain the model. How does the performance change?

We can implement something like

```python
# under __init__ method
self.pos_embed = Embedding(
    seq_len, embed_dim
)
...

# under forward method
# Add positional embedding
positions = torch.arange(seq_len).to(x.device)
hidden = patches + self.pos_emb(positions)
```

In Chapter 11, we will look at other types of embeddings other than sinusoidal positional embeddings. Readers are encouraged to find out more on this topic of research.

