1. List the ways to perform indexing operations.
2. Find the indices of entries for the following array where the values are negative: `[1, -2, 3, -4, -5, -6]`.
3. Suppose we have a 3D tensor cube of shape `(8, 8, 8)`, e.g. `np.arange(512).reshape(8, 8, 8)`. Extract the diagonal entries of the cube.
4. How can you generalize the above exercise so that it can extract the diagonal of tensors of any dimensionality? Assuming the tensor is always a hypercube, with each dimensions having identical size.
5. Apply `get_consecutive_index` to the array `[-3, -2, -1, 1, 2, 5, 8, 9, 10]`, and follow the logic of the illustraion as well as the code step by step.
6. How does `tf.gather` differ from `tf.gather_nd`?
7. Implement `gather_nd` operation using NumPy. Consider converting multi-index to flat index first.
8. Given a square matrix of shape `(5, 5)` (e.g. `np.arange(25).reshape(5, 5)`), set diagonal entries to zeros using NumPy.
9. Using PyTorch to implement a version of the above exercise generalized to multi-dimensional tensors whose diagonal needs to be filled with arbitrary values, which we call `setting_diagonal_to_value`. Use `torch.index_put` operations.
10. Create a checkerboard pattern using put and scatter operations. The odd entries (in terms of flat index) are positive values, the even entries are negative values. For example:

```python
[[ 1,  -1,   2,   -2,    3],
 [-3,   4,  -4,    5,   -5],
 [ 6,  -6,   7,   -7,    8],
 [-8,   9,  -9,   10,  -10],
 [11, -11,  12,  -12,   13]]
```