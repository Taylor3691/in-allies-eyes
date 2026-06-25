## Why

Executing Jaccard re-ranking calculations on the entire 19,732-image gallery database of Market-1501 causes severe performance degradation and browser tab crashes during demos. Restricting the computation to a localized pool of the Top-200 initial matches ensures the frontend remains responsive while still providing the precision improvements of CA-Jaccard.

## What Changes

- Integrate the Context-Aware Jaccard (CA-Jaccard) re-ranking equations into the search retrieval loop.
- Apply CA-Jaccard exclusively to a localized Top-200 subset of Cosine similarity matches.
- Map the query's camera ID and top matches' camera IDs to compute the Jaccard camera mask.

## Capabilities

### New Capabilities
- `integrate-localized-caj-reranking`: Specification for localized CA-Jaccard re-ranking and camera masking requirements.

### Modified Capabilities
<!-- None -->

## Impact

- Modifies [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py)'s search gallery loop.
- Imports [rerank.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/caj/utils/rerank.py) utilities.
