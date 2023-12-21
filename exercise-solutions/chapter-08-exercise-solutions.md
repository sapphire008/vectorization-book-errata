1. How does a jagged tensor differ from a sparse tensor?

A jagged / ragged tensor always assumes a list of lists, whereas sparse tensor does not since it can have empty values / skip entries at any given point. However, sparse tensors can be used to represent ragged tensors.

2. Create a sparse tensor from the following dense tensor, and left align the values.

```python
[   [1, 0, 0, 7],
    [0, 0, 2, 0],
    [0, 0, 0, 0],
    [3, 0, 5, 0]
]
```

```python
import tensorflow as tf
# TODO: import sparse_left_align
X = tf.constant([
    [1, 0, 0, 7],
    [0, 0, 2, 0],
    [0, 0, 0, 0],
    [3, 0, 5, 0]
])

X_sparse = tf.sparse.from_dense(X)
X_sparse = sparse_left_align(X_sparse)

# SparseTensor(indices=tf.Tensor(
# [[0 0]
#  [0 1]
#  [1 0]
#  [3 0]
#  [3 1]], shape=(5, 2), dtype=int64), 
# values=tf.Tensor([1 7 2 3 5], shape=(5,), dtype=int32), 
# dense_shape=tf.Tensor([4 4], shape=(2,), dtype=int64))
```

3. Convert the above sparse tensor into a ragged tensor in Tensorflow, and compute the sum of each row.

```python
# Code continuing from above

X_ragged = tf.RaggedTensor.from_sparse(X_sparse)
tf.reduce_sum(X_ragged, axis=1)

# <tf.Tensor: shape=(4,), dtype=int32, numpy=array([8, 2, 0, 8], dtype=int32)>

```

4. Convert the sparse tensor obtained from Question #2 and convert to a sparse indicator tensor, where the value is 1 if the entry is non-zero, and 0 otherwise. Suppose the vocabulary size is 10.

```python
# Code continuing from above
indicator = tf.sparse.to_indicator(X_sparse, 10)
indicator = tf.cast(indicator, dtype=tf.int32)
```

5. Implement `sparse_to_indicator` in PyTorch.

```python
import torch

import torch

def sparse_to_indicator_torch(
    X: torch.Tensor, # sparse tensor
    vocab_size: int,
    out_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """
    Convert sparse tensor of indices to vectors of binary indicators

    * X: torch.Tensor
        Each row of the sparse matrix is the indices of the indicator vector
    * vocab_size: int
        This determines the size of the sparse indicator vector
    * out_dtype: torch.dtype
        DType of the values of the output tensor
    """
    # create indices of the output sparse tensor
    indices = torch.stack([X.indices()[0, :], X.values()], axis=0)
    # indicator as ones
    values = torch.ones_like(indices[1])
    # (batch_size, vocab_size)
    shape = (X.shape[0], vocab_size)

    # Make the sparse tensor
    y = torch.torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape,
        dtype=out_dtype,
    ).coalesce()

    return y

```

6. Do some research and find out how object detection tasks use IOU or Jaccard Similarity to evaluate model performance.

Object detection tasks usually have ground-truth labels that draws a bounding box around the object to be detected. Models like YOLO directly predict the coordinates of a bounding box to detect objects. Researchers use IOU to compute the overlap between the predicted bounding box from the model and the ground truth label. The higher the IOU metric, the better the model performance is.

7. Use the `set_operation` implementation to compute the set difference of following two sets of batched values:

```python
A = [   
    [1, 2, 3, 0],
    [4, 5, 0, 0],
    [7, 0, 0, 0]
]
B = [
    [1, 0, 0, 0],
    [2, 3, 8, 0],
    [4, 5, 6, 7]
]
```

(i.e. `A - B`) where `0` is a padding value.


```python
import numpy as np
# TODO: import set_operation

A = np.array([
    [1, 2, 3, 0],
    [4, 5, 0, 0],
    [7, 0, 0, 0]
])
B = np.array([
    [1, 0, 0, 0],
    [2, 3, 8, 0],
    [4, 5, 6, 7]
])

diff = set_operation(A, B, pad=0, operation="difference", returns="result")

# diff.A
# array([[2, 3, 0, 0, 0, 0, 0, 0, 0],
#        [4, 5, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0]])
```

8. Use Tensorflow's `tf.sets.difference` function to compute the set difference of the above two sets of batched values (consider converting the two tensors into `tf.SparseTensors`). Compare the results. How much time does converting from dense to sparse tensor take?

```python
import tensorflow as tf

A = tf.constant([
    [1, 2, 3, 0],
    [4, 5, 0, 0],
    [7, 0, 0, 0]
])
B = tf.constant([
    [1, 0, 0, 0],
    [2, 3, 8, 0],
    [4, 5, 6, 7]
])

A = tf.sparse.from_dense(A) # 102 µs ± 3.11 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
B = tf.sparse.from_dense(B) 
diff = tf.sets.difference(A, B)
```

There are some extra time taken to convert from sparse to dense tensor in this case. But for small tensors, the cost is not signficant. Only when we are dealing with large sparse tensors, the memory as well as speed of this operation needs to be carefully considered.


9. Compare and contrast logic of the ragged range operation implemented in `set_operation` function with the one implemented in Chapter 5 `tf_ragged_range`. What are the advantages and disadvantages of each approach?

In Chapter 5's exercises, we have seen that if a sequence is very long, then the shorter sequence needs to be padded with zeros to represent the dense matrix. The advantage of the current implementation is that it does not need to create an intermediate dense matrix which are then masked and compressed. This implementaion is more memory efficient.


10. Consider PyTorch's implementation of `Autoencoder`. Is there a way to leverage `torch.nested.nested_tensor` for our implementation? What are the advantages and disadvantages of using `nested_tensor`?

`torch.nested.nested_tensor` is equivalent to Tensorflow's `tf.RaggedTensor`, though the current implementation uses `tf.SparseTensor` to represent a ragged tensor. Readers can attempt to translate the Tensorflow implementation to a PyTorch implementation. However, more popular way of implementing models with variable length inputs are to pad the sequences to an identical length. As an experimental feature (at the time of writing), `torch.nested.nested_tensor` is less used by the research community currently.
