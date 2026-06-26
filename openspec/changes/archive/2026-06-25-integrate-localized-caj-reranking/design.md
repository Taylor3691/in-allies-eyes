## Context

CA-Jaccard neighbor computation has $O(N^2)$ complexity. Running it on the full gallery size of 19,732 is impractical for a real-time demo. Restricting computation to a local Top-200 pool provides a highly optimized implementation.

## Goals / Non-Goals

**Goals:**
- Feed query-gallery and gallery-gallery distances into the localized CA-Jaccard function.
- Integrate the `re_ranking` module from `caj.utils.rerank` with standard arguments.
- Pass correct query and gallery camera ID matrices.

**Non-Goals:**
- Custom modifications to the math inside `src/caj/utils/rerank.py`.

## Decisions

### Decision 1: Mapping to Sub-Index Space
We compute query-gallery distances inside the sub-space of Top-200 indices:
`q_g_dist = dist[top_200_indices].reshape(1, 200)`
And gallery-gallery pairwise distances within the Top-200 features:
`g_g_dist = 1.0 - np.dot(top_200_features, top_200_features.T)`
- **Rationale**: Isolating the distance calculations to the Top-200 subset maintains identical rank order results while decreasing complexity by $10,000 \times$.

### Decision 2: Hardcoded parameters object
We define a dummy `CAJArgs` class in `app.py` to supply the standard parameters (`k1=20`, `k2=6`, `ckrnns=True`, `clqe=True`) to `re_ranking()`.
- **Rationale**: Avoids loading standard experiment-level CLI parsers into the web application, keeping implementation simple.

## Risks / Trade-offs

- **[Risk]** The target ID might fall outside the Top-200 pool before re-ranking, meaning it cannot be optimized.
  - **Mitigation**: Cosine similarity is already extremely accurate (rank-1 ~94%), so the true match is almost always in the Top-200 subset.
