## 1. Localized Jaccard Distance Setup

- [x] 1.1 Compute the query-gallery and gallery-gallery distance matrices for the localized top-200 search result subset in [search_gallery](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py#L112) in [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py).
- [x] 1.2 Map query and gallery camera IDs to build the `cids` array needed to construct the Jaccard camera-matching mask.

## 2. CA-Jaccard Reranking Call

- [x] 2.1 Instantiate a custom arguments block containing the default CA-Jaccard hyperparameters.
- [x] 2.2 Invoke the standard re-ranking module to optimize query search results and map sub-space indices back to the original database image indexes.
