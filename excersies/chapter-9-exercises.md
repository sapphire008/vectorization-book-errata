1. Use the DataFrame from Section 9.1 to answer the following question: How many total hours of viewership does each movie have? Hint: you might want to use `groupby` and `sum` to answer this question.
2. What is the relationship between broadcasting and groupby operations?
3. Discuss on the disadvantages of `sliding_window_tf` in contrast to PyTorch and NumPy's sliding window operations (consider the concepts of `view` vs. `copy`).
4. Using the idea of bucketizaion and quantization, represent a set of float numbers between `[-1, 1]` (e.g. `[0.62290916, -0.37006572,  0.49293062, -0.87785846,  0.62497933]`) using 8-bit integers (i.e. between -128 and 127, total 256 numbers). What is the average error of this representation?
5. Compare and contrast the three sets of segment-wise operations in Tensorflow.
6. Try to apply the `np_mean_reduceat` on the following data:

    ```python
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    indices = np.array([0, 3, 8])
    ```
    
    What is the result? Is this expected? Try to verify it using a simple Python for-loop.
7. Try to implement the logic of `tf_sparse_segment_prod` using PyTorch with `torch.scatter_add`.
8. Draw a schematic diagram `torch_scatter_reduce_sqrt_n`. Is there an alternative way to implement this function, using `torch.segment_reduce`?
9. Consider the relationships between sparse tensors, jagged tensors, and the segment-wise operations. What are the advantages and disadvantages of each?
10. Various types of positional encoding / embedding schemes have been proposed since the original Transformer paper. One way to encode positions is to use learned embeddings (where the positions are treated as categorical / ordinal features). Try to implement an embedding based positional encodings in PyTorch. Replace the sinusoidal encoding with the learnable embeddings in the ViT model and try to retrain the model. How does the performance change?


