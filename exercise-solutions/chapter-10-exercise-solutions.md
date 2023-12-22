1. `np.sort` implements multiple types of sorting algorithms. Users can choose which algorithm to use by specifying the `kind` argument. Do an experiment and try to sort a large array of random numbers using different sorting algorithms and compare their performance. Which algorithm is the fastest? Which algorithm is the slowest?

We can do the following experiment:

```python
import time
import numpy as np
import pandas
trials = 50
performance = {"quicksort":[], "mergesort":[], "heapsort":[], "stable":[]}
for _ in range(trials):
    # make sequence
    x = np.random.rand(1_000_000)
    for method in performance:
        tnow = time.time()
        np.sort(x, kind=method)
        performance[method].append(time.time() - tnow)
    
df = pd.DataFrame(performance)
df.mean()

# quicksort    0.078864
# mergesort    0.103829
# heapsort     0.110268
# stable       0.104112
# dtype: float64
```
 It appears quicksort is fastest while heapsort is slowest over 50 trials, sorting a random array of 1 million elements.


2. `np.argpartition` uses a bisection algorithm. How is this related to quicksort?

Quicksort also has the concept of pivot. For each iteration, it moves values greater than the pivot value to the right of the pivot, and values less than the pivot value left to the pivot. After all values on the left of the pivot are smaller while values to the right of the pivot greater than the pivot value, the pivot point is served as a point of parition so that the subsequent chunks can then be sorted, with the same logic of determining a pivot and then moving values around the pivot.

3. How could one use `np.argmax` to implement `np.argmin`? (Hint: think about the relationship between `argmax` and `argmin` and the relationship between `max` and `min`.)

One can use `np.argmax` to sort the negative values of the inputs

```python
import numpy as np
x = np.arange(10)
min_index = np.argmax(-x)
```

4. Get top 3 values of the following array `[1, 5, 3, 8, 2, 9, 0, 4, 7]`, with the results positioned in the original order of these list (i.e. expecting `[8, 9, 7]`).

```python
import numpy as np
# TODO: import top_k_partition

X = np.array([1, 5, 3, 8, 2, 9, 0, 4, 7])
top_values, top_indices = top_k_argpartition(X, k=3)
# put in the original ordering
y = -np.ones_like(X)
y[top_indices] = top_values
# masking out the padding values
y = np.compress(y > -1, y)
```

5. `np.unique` can return total 4 outputs, i.e. `unique_values, indices, inverse_indices, counts = np.unique(x, return_index=True, return_inverse=True, return_counts=True)`. Discuss, compare, and contrast these outputs. How can we use them to reconstruct the input?

`unique_values` is the unique set of values from the input, which are usually sorted. `indices` is returned by the flag `return_index=True`, and is the index of the first value encountered along the original list. It has length identical to that of `unique_values`. `inverse_indices` is returend by the flag `return_inverse=True`, and is the reversing index to reconstruct the original inputs using the `unique_values`. It has a length identical to that of the input. `count` is returned by the flag `return_counts=True` and is the count of number of occurrence of each unqiue value. It has shape identical to that of the `unique_values`. We can use the `unique_values` and `inverse_indices` arrays to recontruct the original inputs.


```python
import numpy as np
x = [1, 5, 3, 2, 8, 4, 3, 3, 6]
unique_values, indices, inverse_indices, counts = np.unique(
    x, return_index=True, return_inverse=True, return_counts=True)
x_recov = np.take(unique_values, inverse_indices)
assert len(x) == len(x_recov)
assert np.allclose(x, x_recov)
```

6. Try to sort rows of characters of a 2D matrix by concatenating the characters together and then apply `.sort` function. For example, if the input is `[['h', 'g', 'f'], ['c', 'b', 'a']]`, we first concatenate the characters in each row to get `['cba', 'hgf']` and then apply `.sort` function to get `['cba', 'hgf']`; then split the sorted strings back to characters to get `[['c', 'b', 'a'], ['h', 'g', 'f']]`. How does this approach compare to `lexsort` or `unique` that we implemented?

This concatenation approach would be slower. Let us simulate this using a larger example.

```python
import numpy as np
import pandas as pd
# TODO: import sort_rows_unique
X = np.random.randint(97, 123, size=(3000))
X = [chr(x) for x in X]
X = np.array(X).reshape(1000, 3)

%timeit out = sort_rows_unique(X) # 491 µs ± 5.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# Use Pandas for vectorized string operations
def word_sort(X: pd.Series):
	x_join = X.str.join("-")
	x_join = x_join.sort_values()
	x_out = x_join.str.split("-")
	return np.stack(x_out, axis=0)

X = pd.Series(X.tolist())
%timeit out = word_sort(X) # 1.36 ms ± 25.6 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

```

The `sort_rows_unique` is about twice as fast as the joining-sorting-splitting approach.


7. Reverse the following padded sequence by applying hte `reverse_sequence` function. `[[1, 3, 2, 5, 0, 0, 0, 0], [1, 5, 3, 0, 0, 0, 0, 0], [1, 7, 6, 2, 5, 0, 0, 0]]`. Assume `0` is the padding value. Determine the sequence length `seq_lengths` argument based on the input.


```python
import numpy as np
# TODO: import reverse_sequence
X = np.array([[1, 3, 2, 5, 0, 0, 0, 0], [1, 5, 3, 0, 0, 0, 0, 0], [1, 7, 6, 2, 5, 0, 0, 0]])
seq_lengths = np.sum(X > 0, axis=1)
X_reversed = reverse_sequence(X, seq_lengths)

# array([[5, 2, 3, 1, 0, 0, 0, 0],
#        [3, 5, 1, 0, 0, 0, 0, 0],
#        [5, 2, 6, 7, 1, 0, 0, 0]])
```

8. Consider how we can reverse a `tf.RaggedTensor`, e.g. `tf.ragged.constant([[1, 3, 2, 5], [1, 5, 3], [1, 7, 6, 2, 5]])` (Hint: consider using simple indexing). Can the same idea be applied to `torch.nested.nested_tensor`?

This can be done simply by using slicing:

```python
import tensorflow as tf
X = tf.ragged.constant([[1, 3, 2, 5], [1, 5, 3], [1, 7, 6, 2, 5]])
X_reversed = X[:, ::-1]
```

Apply the same simple slicing logic on `torch.nested.nested_tensor` would throw an `NotImplementedError` at the time of writing, in `torch==2.1.0`.

9. Try to implement `gumbel_max_sample_without_replacement` using NumPy.

```python
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
# TODO: import sparse_to_indicator_scipy from Chapter-8
# TODO: import top_k_argpartition from Chapter-10

def gumbel_max_sample_without_replacement(
    true_labels: coo_matrix,
    num_neg_samples: int,
    vocab_size: int,
    vocab_freq: np.ndarray = None,
    shuffle: bool = False,
):
    """
    Use Gumbel-Max trick to select negative samples.
    * true_labels: (batch_size, None)
        Positive / label class indices, 1-based indexing
    * num_nega_samples:
        Number of negative samples to take
    * vocab_size:
        Number of items in the categorical variable
    * vocab_freq:
        The distribution of the classes to take samples from.
        Does not have to be normalized.
        If None, assuming uniform sampling.
    * shuffle:
        Whether or not to shuffle the results so that the
        positive labels are not always at the beginning of
        the resulting sample.
    """
    # Default to uniform sampling if vocab_freq is None
    if vocab_freq is None:
        vocab_freq = np.ones((vocab_size, ), np.float32)
        
    # Compute proba for the batch
    true_proba = sparse_to_indicator_scipy(true_labels, vocab_size+1).A
    true_proba = true_proba[:, 1:]  # get rid of OOV bucket
    neg_proba = (1.0 - true_proba) * vocab_freq
    
    # Gumbel-Max trick: perturbed logits
    z = -np.log(-np.log(np.random.rand(*neg_proba.shape)))
    z += np.log(neg_proba + 1e-8)
    # set true targets as -inf, so it won't be sampled
    z = np.where(true_proba > 0.5, -np.inf, z)
    
    # Take negative samples 
    # (batch_size, num_neg_samples)
    _, neg_samples = top_k_argpartition(z, k=num_neg_samples)
    
    # offset by 1, since we use 1-based indexing
    neg_samples = neg_samples.astype(np.int32) + 1
    # Concat to make the full sample: positive + negative
    samples = np.concatenate([true_labels.A, -neg_samples], axis=1)
    
    # shuffle
    if shuffle:
        # generating random ordered indices
        r = np.random.rand(*samples.shape)
        r = np.argsort(r, axis=1)
        # Apply the shuffled indices
        samples = np.take_along_axis(samples, r, axis=1)
    
    # Convert to sparse
    samples = coo_matrix(samples)
    
    # Left align
    row_indices = samples.row
    _, row_counts = np.unique(row_indices, return_counts=True)
    indptr = np.concatenate([[0], row_counts])
    indptr = np.cumsum(indptr)
    flat_indices = np.arange(len(samples.data))
    offsets = np.repeat(indptr[:-1], row_counts)
    column_indices = flat_indices - offsets
    
    # Make the matrices
    data = np.where(samples.data > 0, 1.0, 0.0)
    binary_labels = coo_matrix(
        (data, (row_indices, column_indices)), shape=samples.shape,    
    )
    samples = coo_matrix(
        (np.abs(samples.data), (row_indices, column_indices)), shape=samples.shape,    
    )
    
    return samples, binary_labels

# testing out the implementation
true_labels = coo_matrix(
    np.array([[1, 4, 0], [2, 3, 6], [8, 0, 0], [7, 2, 0]])
)

samples, binary_labels = gumbel_max_sample_without_replacement(
    true_labels=true_labels,
    num_neg_samples=3,
    vocab_size=10,
    vocab_freq= None,
    shuffle=True,
)

# samples.A
# array([[ 6,  1,  4,  2,  5,  0],
#        [ 4, 10,  7,  6,  3,  2],
#        [ 6,  9,  4,  8,  0,  0],
#        [ 2,  6,  3,  1,  7,  0]])

# binary_labels.A
# array([[0., 1., 1., 0., 0., 0.],
#        [0., 0., 0., 1., 1., 1.],
#        [0., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0.]])
```

10. Read the appendix section from the cited reference paper "Review of the Gumbel-max Trick and its Extensions for Discrete Stochasticity in Machine Learning" (http://arxiv.org/abs/ 2110.01515 (2022)), which provided a detailed derivation of the Gumbel-max trick.

The appendix provides a detials mathematical derivation of the Gumbel-max trick that we implemented in this chapter.