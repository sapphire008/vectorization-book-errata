1. List the ways to perform indexing operations.

* Integer indexing with flat or multi-index
* Boolean indexing using arrays of True and False, where True values will be retained

2. Find the indices of entries for the following array where the values are negative: `[1, -2, 3, -4, -5, -6]`.

```python
import numpy as np
arr = np.array([1, -2, 3, -4, -5, -6])
index = np.where(arr < 0)[0]
```

3. Suppose we have a 3D tensor cube of shape `(8, 8, 8)`, e.g. `np.arange(512).reshape(8, 8, 8)`. Extract the diagonal entries of the cube.

```python
import numpy as np
tensor = np.arange(512).reshape(8, 8, 8)
diag_index = np.arange(8)
diag_entries = tensor[diag_index, diag_index, diag_index]
```

4. How can you generalize the above exercise so that it can extract the diagonal of tensors of any dimensionality? Assuming the tensor is always a hypercube, with each dimensions having identical size.

```python
import numpy as np

def get_diagonal(tensor):
	# numebr of entries in the diagonal
	n = tensor.shape[0]
	diag_index = np.arange(n)
	diag_index = np.vstack([diag_index] * tensor.ndim)
	diag_flat_index = np.ravel_multi_index(diag_index, tensor.shape)
	return np.take(tensor, diag_flat_index)

```

5. Apply `get_consecutive_index` to the array `[-3, -2, -1, 1, 2, 5, 8, 9, 10]`, and follow the logic of the illustraion as well as the code step by step.

```python
import numpy as np
# TODO: import get_consecutive_index
arr = np.array([-3, -2, -1, 1, 2, 5, 8, 9, 10])
res = get_consecutive_index(arr)
# array([[0, 2],
#        [3, 4],
#        [6, 8]])
```

6. How does `tf.gather` differ from `tf.gather_nd`?

`tf.gather` uses flat index or batched indexing along a dimension, while `tf.gather_nd` uses multi-index.

7. Implement `gather_nd` operation using NumPy. Consider converting multi-index to flat index first.

```python
import numpy as np

def gather_nd(tensor, index):
	flat_index = np.ravel_multi_index(index)
	return np.take(tensor, flat_index)
```

8. Given a square matrix of shape `(5, 5)` (e.g. `np.arange(25).reshape(5, 5)`), set diagonal entries to zeros using NumPy.

```python
import numpy as np
arr = np.arange(25).reshape(5, 5)
np.fill_diagonal(arr, 0)
```

9. Using PyTorch to implement a version of the above exercise generalized to multi-dimensional tensors whose diagonal needs to be filled with arbitrary values, which we call `setting_diagonal_to_value`. Use `torch.index_put` operations.

```python
import torch

def setting_diagonal_to_value(tensor, fill_value=0.):
	# Get diagonal entry indices
	n = min(tensor.shape)
	diag_index = torch.arange(n)
	fill_value = torch.tensor(fill_value, dtype=tensor.dtype)
	tensor = tensor.index_put_([diag_index] * n, fill_value, accumulate=False)
	return tensor

```


10. Create a checkerboard pattern using put and scatter operations. The odd entries (in terms of flat index) are positive values, the even entries are negative values. For example:

```python
[[ 1,  -1,   2,   -2,    3],
 [-3,   4,  -4,    5,   -5],
 [ 6,  -6,   7,   -7,    8],
 [-8,   9,  -9,   10,  -10],
 [11, -11,  12,  -12,   13]]
```


```python
import numpy as np

def checkerboard_pattern(n):
	n_values = n**2
	pos_values = np.arange(n_values//2 + n_values%2) + 1
	neg_values = np.arange(n_values//2) + 1
	neg_values = -neg_values
	out = np.zeros(n_values, dtype=int)
	out[::2] = pos_values
	out[1::2] = neg_values
	out = out.reshape(n, n)
	return out
```

