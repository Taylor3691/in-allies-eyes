## Context

The scientific workbench requires comparing raw webcam frame crops with Kalman-smoothed crops. We process both crops against the precomputed gallery, running search loops with CA-Jaccard enabled and disabled.

## Goals / Non-Goals

**Goals:**
- Extract query features from both input crops.
- Run baseline search and CA-Jaccard re-ranking on both features.
- Return top-5 results arrays for all four modes.

**Non-Goals:**
- Allowing different dataset selectors (the workbench is locked to Market-1501).

## Decisions

### Decision 1: Reuse search gallery logic
We implement `run_comparison()` by calling `search_gallery()` four times with different arguments:
- Mode 1: `search_gallery(raw_img, use_caj=False, top_k=5)` -> baseline output
- Mode 2: `search_gallery(kalman_img, use_caj=False, top_k=5)` -> baseline output
- Mode 3: `search_gallery(raw_img, use_caj=True, top_k=5)` -> final output
- Mode 4: `search_gallery(kalman_img, use_caj=True, top_k=5)` -> final output
- **Rationale**: Reusing the same search function guarantees identical feature extraction, distance computation, and rendering properties across the workbench and Tab 2, avoiding code redundancy.

## Risks / Trade-offs

- **[Risk]** Running four searches sequentially might block the main thread for 8-10 seconds on first execution.
  - **Mitigation**: The Re-ID model and gallery cache are loaded lazily and cached globally. By the time the user opens Tab 3, the cache and model are already loaded, dropping sequential run-time to under 0.2 seconds.
