1. How does Hadamard Product (element-wise product) differ from Matrix Multiplication?
2. Implement matrix multiplication between a matrix of size `(5, 4)` and a matrix of size `(4, 3)` using for-loop. What is the time complexity of this implementation? Compare the performance of this implementation with NumPy's `np.matmul` function.
3. Suppose we have the following matrices

```python
import numpy as np
import torch
A = np.arange(20).reshape(5, 4)
B = torch.arange(12).reshape(4, 3)
```

What would happen in the operation `A @ B`?
4. Use `einsum` to implement a tensor multiplication between a tensor of shape `(5, 4, 3)` and a tensor of shape `(3, 4, 2)`, so that the resulting shape of the output is `(5, 4, 4, 2)`. Here, we treat the first dimension of the first tensor as the batch dimension, and the last dimension of the second tensor as the batch dimension as well. That is, we would like to do batch-wise matrix multiplication between pairs of matrices of shape `(4, 3)` and `(3, 4)`.
5. Implement the same operations as above using `tensordot` instead.
6. Compare the performance of `faster_cross_corr` with `np.corrcoef`. Plot the time taken for each function as a function of the size of the input.
7. What is the difference between inverse and Penrose-Moore pseudo-inverse? Illustrate the difference in results given a random square matrix to be inverted.
8. Use Singular Value Decomposition (SVD) to perform Principal Component Analysis (PCA) on the following data points. Plot the main axes of the components.

```python
import numpy as np
x  = np.random.multivariate_normal(
    mean=[-1, 1], 
    cov=[[0.2, 0.6], [0.6, 0.2]], 
    size=5000
)
```

9. Implement double-exponential curve fitting as described by the book "Regressions et Equations Integrale".
10. Try to fit the following data points using the double-exponential curve fitting method.

```python
import numpy as np
x = np.random.rand(500) * 10
x = np.sort(x)
Y = 0.2 * np.exp(-3 * x) + 0.9 * np.exp(-0.5 * x) + 0.05 * np.random.randn(500)
```