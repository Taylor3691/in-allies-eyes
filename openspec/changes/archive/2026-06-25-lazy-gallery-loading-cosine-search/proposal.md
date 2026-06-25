## Why

Loading the 19,732 gallery image embeddings into memory at application boot degrades start-up times. To keep launch instantaneous and search fast, we need to load the gallery cache lazily upon the first query search and perform a quick Cosine similarity distance computation to establish baseline ranking.

## What Changes

- Implement lazy loading for the precomputed gallery cache file `market1501_gallery_features.npy`.
- Implement a fast Cosine similarity comparison between the query embedding and all loaded gallery embeddings.
- Add baseline top-K results rendering to the Gradio gallery component.

## Capabilities

### New Capabilities
- `lazy-gallery-loading-cosine-search`: Specifications of requirements for lazy cache loading and Cosine similarity search matching.

### Modified Capabilities
<!-- None -->

## Impact

- Modifies [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py)'s gallery search callback.
- Reads `market1501_gallery_features.npy` cache file from disk.
