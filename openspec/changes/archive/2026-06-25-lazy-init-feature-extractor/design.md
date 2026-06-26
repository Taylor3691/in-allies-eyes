## Context

Deep learning checkpoints for person Re-ID ResNet-50 are large (approx. 100-200MB) and take several seconds to load into memory. Loading them at app startup degrades user experience. Deferring the loading until actually needed (lazy initialization) ensures a fast startup (<1s) for the Gradio frontend.

## Goals / Non-Goals

**Goals:**
- Implement lazy loading for the Re-ID feature extractor model.
- Cache the model instance globally in memory to avoid reloading overhead.
- Integrate the model loading into the webcam capture and search gallery loops.

**Non-Goals:**
- Loading weights dynamically over the network (checkpoints are already stored in `pretrained_models/`).
- Lazy loading dataset images (we already precomputed the gallery features).

## Decisions

### Decision 1: Thread-safe lazy model loading
We will use a helper function/class in `app.py` that checks if a global `reid_model` variable is `None`. If it is `None`, it loads the model and state dict.
- **Rationale**: Keeps model instantiation logic simple and avoids concurrent initialization threads if clicks are throttled properly.

### Decision 2: Cached PyTorch weights path
We default the checkpoint loading path to `pretrained_models/market_resnet50_model_120_rank1_945.pth` (which is a BoT model).
- **Rationale**: This matches the model we used to generate the precomputed gallery embeddings, ensuring query and gallery feature spaces are identical.

## Risks / Trade-offs

- **[Risk]** First search/capture query will experience a 2-3 second delay.
  - **Mitigation**: Add a Gradio loading indicator (spinner) to notify the user that model loading is in progress.
