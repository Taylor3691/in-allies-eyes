## Context

The Market-1501 dataset contains thousands of gallery images. Loading all gallery embeddings at launch increases memory usage and app boot latency. Deferring the disk read of `market1501_gallery_features.npy` until the first query search avoids startup lag.

## Goals / Non-Goals

**Goals:**
- Implement lazy loading of the `.npy` gallery embedding database.
- Cache the loaded embeddings globally in memory for all subsequent searches.
- Compare query and gallery embeddings using normalized Cosine similarity distance.

**Non-Goals:**
- Creating custom feature normalization logic (the query features should be normalized directly by the lazy loader's model outputs).

## Decisions

### Decision 1: NumPy cache file mapping
We load the precomputed gallery dictionary (`features`, `image_paths`, `camids`, `pids`) using `np.load()` with `allow_pickle=True`.
- **Rationale**: Storing the values in a single `.npy` file makes read operations simple and fast on standard storage hardware.

### Decision 2: Cosine Similarity computed via matrix multiplication
We compute distances as: `dist = 1.0 - np.dot(query_feat, g_features.T)` after L2 normalizing both arrays.
- **Rationale**: Computing the dot product is computationally trivial and runs in under 1ms on CPU for 19,000+ gallery features, eliminating potential bottleneck concerns.

## Risks / Trade-offs

- **[Risk]** The cache `.npy` file might be missing or corrupt.
  - **Mitigation**: Perform a file existence check and return a clean Gradio placeholder message/image if the file is not found.
