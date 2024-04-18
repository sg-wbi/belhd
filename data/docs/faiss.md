# Brute force

When the number of query vectors is limited,
the best indexing method is to not index at all and use brute force search,
since the index building time will not be amortized by the search time.
Use `faiss.knn` or `faiss.knn_gpu`.
You can pass directly pytorch tensors by `import faiss.contrib.torch_utils` (see [here](https://github.com/facebookresearch/faiss/blob/main/contrib/torch_utils.py) for more details).
Direct computation can be done via a "Flat" index.

## Without faiss:

[numpy](https://gist.github.com/mdouze/a8c914eb8c5c8306194ea1da48a577d2): no faiss installation needed

```python
def np_knn(xq, xb, k): 
    # knn function in numpy. This mimics closely what is computed in Faiss 
    # without the tiling (will OOM with too large matrices)
    norms_xq = (xq ** 2).sum(axis=1)
    norms_xb = (xb ** 2).sum(axis=1)
    distances = norms_xq.reshape(-1, 1) + norms_xb -2 * xq @ xb.T 
    I = np.argpartition(distances, k, axis=1)[:, :k]
    D = np.take_along_axis(distances, I, axis=1)
    # unfortunately argparition does not sort the partition, so need another 
    # round of sorting
    o = np.argsort(D, axis=1)
    return np.take_along_axis(D, o, axis=1), np.take_along_axis(I, o, axis=1)
```

[pytorch](https://gist.github.com/mdouze/551ef6fa0722f2acf58fa2c6fce732d6#file-demo_pytorch_knn-ipynb): collects gradients.

```
def torch_knn(xq, xb, k): 
    # knn function in pytorch. This mimics closely what is computed in Faiss 
    # without the tiling (will OOM with too large matrices)
    norms_xq = (xq ** 2).sum(axis=1)
    norms_xb = (xb ** 2).sum(axis=1)
    distances = norms_xq.reshape(-1, 1) + norms_xb -2 * xq @ xb.T 
    return torch.topk(distances, k, largest=False)
```

# Out-of-memory

If the whole dataset does not fit in RAM, you can build small indexes one after another,
and combine the search results via `faiss.ResultHeap`
(see [here](https://github.com/facebookresearch/faiss/blob/main/faiss/python/extra_wrappers.py#L218) for more details).
