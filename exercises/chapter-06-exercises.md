1. Suppose we have a NumPy array of strings, `A = np.array(["1", "12", "123"])`. The dtype of the array is then `<U3`. If we add one more element `"1234"` to the end of the array, what would the dtype of the array be?
2. Try to run `pd.Series(["a", "b", "c", "d", "e"], dtype="my_dtype")`. What is going to happen?
3. Try using Python string method `.decode("utf")` to convert bytes to string. For example, `np.array([b"hello", b"world"])`. How does the performance compare to `.astype(str)` if the list is very long, e.g. `np.array([b"hello", b"world"]*10000)`?
4. Apply `pd.to_numeric` to the following data: `["1", "2", "a", "b", "8"]`. What are the results. What if we set the argument `errors="coerce"`?
5. Implement the following program using Tensorflow's string processing routines: Split the sentence based on space, then lower-casing each word.

```
However, deep learning library like Tensorflow does not have a built-in timestamp class, so it is more typical to handle timestamp features using strings when at- tempting to model time series data.
```

6. Apply `_parse_timestamps` to the following string timestamps

```python
timestamps_array = [
    "2020-01-01 00:00:05+03:00",
    "2019-12-31 15:08:01-05:00",
    "2028-02-28 23:59:59+04:00",
]
```

7. Use `tf.lookup` to create a hashtable that converts the following features into their corresponding integer values: `["a", "c", "A", "B", "f", "G", "E", "h", "T"]`.

8. Using regular expression, extract how many epochs the model has been trained so far from the log.

```
Epoch 8/10
```

9. Using regular expression (regex), remove any "a" and "the" in the following sentence.

```
Within the context of machine learning data management, serialization is the process of converting structured data into a stream of byte strings so it can be further compressed and consumed by any downstream processes
```

10. Serialize the following data dictionary using Tensorflow's `tf.train.Example` protobuf.
```python
inputs = {
    "user_id": "1",
    "item_id": 12,
    "percent_watched": 0.82,
}
```