1. What are the differences between a sparse matrix/tensor and a dense matrix/tensor?
2. Given the following row, col, and data arrays, create a `csr_matrix` (Hint: starts with a `coo_matrix`):

```python
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 1, 3, 4, 6])
data = np.array([1, 2, 3, 4, 5, 6])
shape = (4, 8)
```

3. Compare memory usage of the dense and sparse representations of the following matrix:

```python
A = np.random.randint(1, 10, size=(1000, 1000), dtype=int)
A[A < 8] = 0
A_sp = coo_matrix(A)
```

To find out the total bytes of a NumPy array, use `A.nbytes`. Similarly, to find out the total bytes of a sparse matrix, use `A_sp.data.nbytes + A_sp.row.nbytes + A_sp.col.nbytes`.

4. Compare the memory usage of the dense and sparse representation of the following matrix

```python
B = np.random.randint(1, 10, size=(1000, 1000), dtype=int)
B[B < 1] = 0
B_sp = coo_matrix(B)
```

Together with Question #3, readers can get a sense when sparse matrix is (or is not) more memory efficient than dense matrix.

5. Min-max normalize (to values between 0 and 1) the rows of the following sparse matrix, given its dense data (0s as the sparse entries; Hint: using column-wise broadcasting)

```python
X = np.array([
    [1, 2, 0, 3, 0],
    [0, 2, 0, 1, 0],
    [2, 5, 3, 4, 0],
    [0, 3, 0, 0, 1],
    [0, 1, 0, 0, 2]
])

```

6. Write a function that can convert Scipy's `coo_matrix` into `tf.SparseTensor`.
7. Use `tf.sparse.reduce_max` to find out the maximum value of each row of the following sparse tensor, given in its dense representation (0s as the sparse entries; use `tf.sparse.from_dense` to convert the dense representation to sparse representation`):

```python
X = tf.constant([
    [1, 2, 0, 3, 0],
    [0, 2, 0, 1, 0],
    [2, 5, 3, 4, 0],
    [0, 3, 0, 0, 1],
    [0, 1, 0, 0, 2]
])
```

8. Do some research and find out what Compressed Sparse Fiber format is (implemented by PyTorch).
9. Tune Alternating Least Squares (ALS) model on the MovieLens dataset. How is the latent feature size `K` and regularization term `l2_reg` trading off?
10. Use ALS for image completion. The reader can start with the following code snippet:

```python
import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix, coo_matrix, dok_matrix, load_npz, save_npz
from tqdm import tqdm
from pdb import set_trace

import os
import skimage # pip install scikit-image==0.22.0
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd

X = skimage.data.camera()
R = X.flatten()
# randomly corrupt the image
rs = np.random.RandomState(42)
index = rs.choice(len(R), size=int(len(R)*0.25), replace=False)
R[index]=0
R = R.reshape([512, 512])
fig, axs = plt.subplots(1, 2, figsize=(10,4))
ax = axs[0]
ax.imshow(X, cmap='gray', interpolation='none')
ax.set_title("Original")
ax = axs[1]
ax.imshow(R, cmap='gray', interpolation='none')
ax.set_title("Corrupted")
```