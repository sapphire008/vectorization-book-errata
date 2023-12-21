1. Implement one-hot encoding in NumPy using `np.eye`, given the inputs `num_values` and `index`.
2. Construct a upper triangular mask that is two entries above the diagonal of a 7x7 matrix.
3. Compare the performance of boolean vs. multiplication method of applying a mask to a set the diagonal entries of a 2D matrix to zero. Let `X` be a large square matrix and `mask` be the non-diagonal entries of the matrix, which can either be a boolean value or a 0/1 value. Use the `time` package to compare the performance of `X[~mask] = 0` vs. `X * mask` vs. `np.where(mask, X, 0)`. Make a plot of the matrix size vs. the time it takes to apply the mask.
4. Implement `XOR` using combinations of `AND` (`&`), `OR` (`|`) and `NOT` (`~`), i.e. `a ^ b = (a | b) & ~(a & b)`, given `a` and `b` are boolean tensors with identical shape.
5. Use `combine_mask` to merge the following two masks with `OR` operation.

```python
mask1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
mask2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
```

6. Compute the average of the top-3 values of each row of the following matrix: `[[0.6, 0.3, 0.7, 0.9, 0.8, 0.4], [0.1, 0.2, 0.3, 0.5, 0.4, 0.8]]` (Hint: use `top_k_masking` to mask out all but the top 3 values, then sum together each row and divide by 3).
7. Pad the rows of the following matrix with average of the row using NumPy's `np.pad` method, to 5 entries. `[[1, 5, 2, 3], [6, 3, 8, 7]]`.
8. Apply `align_lengths` to the following data

```python
X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
```

and

```python
X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([1, 2], [1, 2])
```

9. Apply softmax to the array `[0.5, 0.8, -1e9, 0.7]`. What is the value of the third entry after softmax activation? Consider why we need to use a large negative number as the masked value, instead of setting the masked value to zero after softmax. Discuss pros and cons of each approach.
10. Consider the function `ragged_range` that we implemented in this chapter. Suppose the input `n = [5, 2, 3, 1000]`, where the last array has length significantly greater than that of the rest. What would happen in this case? What is the maximum memory that would be used when creating the ragged range array?
