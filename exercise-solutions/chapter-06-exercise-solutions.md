1. Suppose we have a NumPy array of strings, `A = np.array(["1", "12", "123"])`. The dtype of the array is then `<U3`. If we add one more element `"1234"` to the end of the array, what would the dtype of the array be?

The dtype will become `<U4` since the value following `U` will defer to the longest string.

2. Try to run `pd.Series(["a", "b", "c", "d", "e"], dtype="my_dtype")`. What is going to happen?

This will throw an error. In `pandas==2.1.1`, the error is `TypeError: data type 'my_dtype' not understood`.

3. Try using Python string method `.decode("utf")` to convert bytes to string. For example, `np.array([b"hello", b"world"])`. How does the performance compare to `.astype(str)` if the list is very long, e.g. `np.array([b"hello", b"world"]*10000)`?

```python
import numpy as np
arr = np.array([b"hello", b"world"]*10000)
%timeit [a.decode("utf") for a in arr] # 7.63 ms ± 303 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
%timeit arr.astype(str) # 1.48 ms ± 32 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

Vecotrized method implemented by NumPy is faster than list comprehension with for-loop in pure Python.

4. Apply `pd.to_numeric` to the following data: `["1", "2", "a", "b", "8"]`. What are the results. What if we set the argument `errors="coerce"`?

Calling `pd.to_numeric` will throw an error. In `pandas=2.1.1`, the error message is: `ValueError: Unable to parse string "a" at position 2`. If we set `errors="coerce"`, this will return an array where the values not convertible to numerical values are set as `nan`, i.e. `array([ 1.,  2., nan, nan,  8.])`.

5. Implement the following program using Tensorflow's string processing routines: Split the sentence based on space, then lower-casing each word.

```
However, deep learning library like Tensorflow does not have a built-in timestamp class, so it is more typical to handle timestamp features using strings when at- tempting to model time series data.
```

```python
import tensorflow as tf
sentence = "However, deep learning library like Tensorflow does not have a built-in timestamp class, so it is more typical to handle timestamp features using strings when at- tempting to model time series data."
tokens = tf.strings.split(sentence, " ")
tokens = tf.strings.lower(tokens)

# <tf.Tensor: shape=(32,), dtype=string, numpy=
# array([b'however,', b'deep', b'learning', b'library', b'like',
#        b'tensorflow', b'does', b'not', b'have', b'a', b'built-in',
#        b'timestamp', b'class,', b'so', b'it', b'is', b'more', b'typical',
#        b'to', b'handle', b'timestamp', b'features', b'using', b'strings',
#        b'when', b'at-', b'tempting', b'to', b'model', b'time', b'series',
#       b'data.'], dtype=object)>
```

6. Apply `_parse_timestamps` to the following string timestamps

```python
timestamps_array = [
    "2020-01-01 00:00:05+03:00",
    "2019-12-31 15:08:01-05:00",
    "2028-02-28 23:59:59+04:00",
]
```

```python
import tensorflow as tf
# TODO: import _parse_timestamps

timestamps_array = [
    "2020-01-01 00:00:05+03:00",
    "2019-12-31 15:08:01-05:00",
    "2028-02-28 23:59:59+04:00",
]
outputs = _parse_timestamps(timestamps_array)
outputs = tf.transpose(tf.stack(outputs, axis=0))

# <tf.Tensor: shape=(3, 7), dtype=int64, numpy=
# array([[2020,    1,    1,    0,    0,    5,    3],
#        [2019,   12,   31,   15,    8,    1,   -5],
#       [2028,    2,   28,   23,   59,   59,    4]])>
```


7. Use `tf.lookup` to create a hashtable that converts the following features into their corresponding integer values: `["a", "c", "A", "B", "f", "G", "E", "h", "T"]`.

```python
import tensorflow as tf
x = tf.constant(["a", "c", "A", "B", "f", "G", "E", "h", "T"])
vocab, index = tf.unique(x)

table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(vocab),
        values=tf.constant(index),
    ),
    default_value=-1,
)

val = table.lookup(x)
# <tf.Tensor: shape=(9,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)>
```

8. Using regular expression, extract how many epochs the model has been trained so far from the log.

```
Epoch 8/10
```

```python
import re
# extract the epoch/total_epoch pattern
epochs = re.findall(r'(\d+)/(\d+)', "Epoch 8/10")[0]
# The former number is the number of epochs trained so far
epoch_so_far = int(epochs[0])
```

9. Using regular expression (regex), remove any "a" and "the" in the following sentence.

```
Within the context of machine learning data management, serialization is the process of converting structured data into a stream of byte strings so it can be further compressed and consumed by any downstream processes
```

```python
import re
re.sub(r'\sa|\sthe', "", "Within the context of machine learning data management, serialization is the process of converting structured data into a stream of byte strings so it can be further compressed and consumed by any downstream processes")
```

Here we are also removing the extra spaces around these words.

10. Serialize the following data dictionary using Tensorflow's `tf.train.Example` protobuf.
```python
inputs = {
    "user_id": "1",
    "item_id": 12,
    "percent_watched": 0.82,
}
```


```python
import tensorflow as tf

inputs = {
    "user_id": "1",
    "item_id": 12,
    "percent_watched": 0.82,
}

feature = {
    "user_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(inputs["user_id"], "utf-8")])),
    "item_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[inputs["item_id"]])),
    "percent_watched": tf.train.Feature(float_list=tf.train.FloatList(value=[inputs["percent_watched"]])),
}

example = tf.train.Example(features=tf.train.Features(feature=feature))
serialized_example = example.SerializeToString()
# b'\nA\n\x10\n\x07item_id\x12\x05\x1a\x03\n\x01\x0c\n\x1b\n\x0fpercent_watched\x12\x08\x12\x06\n\x04\x85\xebQ?\n\x10\n\x07user_id\x12\x05\n\x03\n\x011'
```