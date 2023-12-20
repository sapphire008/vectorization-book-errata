1. Differentiate between the following terms: scalar, array, vector, matrix, tensor.
2. Use `np.ones` and `np.full` to construct a 2D matrix filled with `5`, shape `(3, 4)`.
3. Use `np.arange` to construct a vector of consecutive numbers from `0` to `1` including `1`, with increment `0.1`.
4. Use `np.linspace` to construct a vector of consecutive numbers from `0` to `1` including `1`, with `11` elements. How does this differ from the previous exercise?
5. Use `np.random.rand` to generate 1000 numbers and plot the histogram. What does the histogram look like? What happens if you increase the number of random numbers?
6. Apply `np.sum` and `np.nansum` to the array `[1, np.nan, 2, 5, 4, 8]`. How does the result differ?
7. Try to use `np.flatten` and `np.ravel` to flatten a large array, say `np.random.rand(100, 100, 100)`. Compare their performance, and try to explain why the difference in performance.
8. Use the case study on Image Normalization as a template and try to implement a function that normalizes a batch of images to have zero mean and unit variance.
9. Create a random matrix of shape `(4, 7)`, whose values are drawn from the uniform distribution. Then divide each row of the matrix by the mean of the row (i.e. using broadcasting).
10. Construct a magic square of order 12, and plot out the matrix using Matplotlib.



