## Context

Evaluators need to easily inspect the behavior of camera bias and CA-Jaccard rank adjustments. We overlay colored borders directly onto the image tensors returned to Gradio, which is highly reliable.

## Goals / Non-Goals

**Goals:**
- Apply visual borders on matching image arrays prior to rendering.
- Dynamically format captions to summarize rank, camera ID, and rank changes.

**Non-Goals:**
- Injecting raw HTML or style tags into Gradio elements (which can break responsive layout).

## Decisions

### Decision 1: Drawing borders via copyMakeBorder
We draw constant-value borders on the OpenCV image arrays:
`cv2.copyMakeBorder(img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=color_rgb)`
- **Rationale**: Extends the canvas border in the designated color without overlaying or cropping the actual person Re-ID pixels, preserving visibility.

### Decision 2: Captions string formatting
We set captions as: `Rank {idx} | Cam {cam_id} [Bias/Improvement tags]`
- **Rationale**: Standardizes text metadata display directly inside Gradio's image gallery container.

## Risks / Trade-offs

- **[Risk]** Image borders increase overall output dimensions slightly.
  - **Mitigation**: Gradio automatically fits images inside its flex grid slots, so small increases in dimension do not break layout.
