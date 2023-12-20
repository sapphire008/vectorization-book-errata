1. Differentiate between the following terms: scalar, array, vector, matrix, tensor.

* **scalar**, then the variable has 0 dimensions and represents a single numerical value. 
* **array**, or **vector**, then the variable has 1 dimension and represents a list of values.
* **matrix**, then the variable has 2 dimensions and represents a typical 2-dimensional matrix.
* **tensor**, then the variable is a generic collection of values of $N$-dimension(s), where $N \in \{0, 1, 2, ....\}$ is a natural number. More often, I will use the term "tensor" to indicate the high-dimensional collection of values where $N > 2$.

2. Use `np.ones` and `np.full` to construct a 2D matrix filled with `5`, shape `(3, 4)`.

```python
import numpy as np
arr_ones = np.ones((3, 4)) * 5
arr_full = np.full((3, 4), 5)
```

3. Use `np.arange` to construct a vector of consecutive numbers from `0` to `1` including `1`, with increment `0.1`.

```python
import numpy as np
arr = np.arange(0, 1.1, 0.1)
```

4. Use `np.linspace` to construct a vector of consecutive numbers from `0` to `1` including `1`, with `11` elements. How does this differ from the previous exercise?

```python
import numpy as np
arr = np.linspace(0, 1, 11)
```

This results identical array as the previous exercise. When using `np.arange`, we need to know the start, end and the step increment of the array. Here the end value is not included, so if we would like to include the end value itslef, we would need to increase the end value by the amount of step, in this case 1.0 + 0.1 = 1.1. However, using `linspace`, the end value is included, and we need to determine the number of values between the start and end values instead.

5. Use `np.random.rand` to generate 1000 numbers and plot the histogram. What does the histogram look like? What happens if you increase the number of random numbers?

```python
import numpy as np
import matplotlib.pyplot as plt
arr = np.random.rand(1000)
fig, ax = plt.subplots(1, 1)
ax.hist(arr)
```

The histogram looks like a uniform distribution between 0 and 1, but has some unevenness due to noise of sampling. If we increase the nubmer of random numbers, we will see less noise.


6. Apply `np.sum` and `np.nansum` to the array `[1, np.nan, 2, 5, 4, 8]`. How does the result differ?

`np.sum(arr)` will result in `np.nan`, while `np.nansum(arr)` will result in the sum of the non-NaN values, which is `1 + 2 + 5 + 4 + 8 = 20`.

7. Try to use `np.ndarray.flatten` and `np.ravel` to flatten a large array, say `np.random.rand(100, 100, 100)`. Compare their performance, and try to explain why the difference in performance.

```python
import numpy as np
arr = np.random.rand(100, 100, 100)
%timeit arr.flatten() # 757 µs ± 162 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
%timeit np.ravel(arr) # 424 ns ± 25.3 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
```
`np.ndarray.flatten` is slower than `np.ravel`. This is because `np.ndarray.flatten` returns a copy of the flattened tensor, while `np.ravel` returns a view, without actually copying the underlying data.


8. Use the case study on Image Normalization as a template and try to implement a function that normalizes a batch of images to have zero mean and unit variance.

```python
import numpy as np

def standardize_image_batch(images):
	"""Normalize image pixel values to be between 0 and 1"""
	# image shape (batch_size, height, width, channel)
	# mean / std (batch_size, 1, 1, 1)
	mean_val = np.mean(images, axis=(1, 2, 3), keepdims=True)
	std_val = np.std(images, axis=(1, 2, 3), keepdims=True)
	standardized_img = (images - mean_val) / (std_val + 1e-6)
	
	return standardized_img
```

9. Create a random matrix of shape `(4, 7)`, whose values are drawn from the uniform distribution. Then divide each row of the matrix by the mean of the row (i.e. using broadcasting).

```python
import numpy as np
arr = np.random.rand(4, 7)
row_mean = arr.mean(axis=1, keepdims=True) # (4, 1)
arr = arr / row_mean
```

10. Construct a magic square of order 12, and plot out the matrix using Matplotlib.

```python
import numpy as np
import matpotlib.pyplot as plt
# TODO: import magic function

magic_12 = magic(12)

fig, ax = plt.subplots(1, 1)
ax.imshow(magic_12)
```

For a better plot, we can use `heatmap` from the `seaborn==0.13.0` package.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# TODO: import magic function

magic_12 = magic(12)

fig, ax = plt.subplots(1, 1)
sns.heatmap(
	magic_12, vmin=1, vmax=12**2, annot=True, square=True, ax=ax
)

```

