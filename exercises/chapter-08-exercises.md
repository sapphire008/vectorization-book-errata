1. How does a jagged tensor differ from a sparse tensor?
2. Create a sparse tensor from the following dense tensor, and left align the values.

```python
[   [1, 0, 0, 7],
    [0, 0, 2, 0],
    [0, 0, 0, 0],
    [3, 0, 5, 0]
]
```

3. Convert the above sparse tensor into a ragged tensor in Tensorflow, and compute the sum of each row.
4. Convert the sparse tensor obtained from Question #2 and convert to a sparse indicator tensor, where the value is 1 if the entry is non-zero, and 0 otherwise. Suppose the vocabulary size is 10.
5. Implement `sparse_to_indicator` in PyTorch.
6. Do some research and find out how object detection tasks use IOU or Jaccard Similarity to evaluate model performance.
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
8. Use Tensorflow's `tf.sets.difference` function to compute the set difference of the above two sets of batched values (consider converting the two tensors into `tf.SparseTensors`). Compare the results. How much time does converting from dense to sparse tensor take?
9. Compare and contrast logic of the ragged range operation implemented in `set_operation` function with the one implemented in Chapter 5 `tf_ragged_range`. What are the advantages and disadvantages of each approach?
10. Consider PyTorch's implementation of `Autoencoder`. Is there a way to leverage `torch.nested.nested_tensor` for our implementation? What are the advantages and disadvantages of using `nested_tensor`?