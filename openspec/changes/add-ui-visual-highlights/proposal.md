## Why

When evaluating Re-ID searches, users need immediate visual feedback to distinguish between baseline search results and optimized results, and to spot camera bias patterns. Drawing prominent color outlines and embedding descriptive rank details directly on the match images provides an intuitive visual verification of search success and CA-Jaccard correction effectiveness.

## What Changes

- Highlight matching gallery images from the same camera as the query using a Yellow border overlay and same-camera label.
- Highlight gallery images whose ranking position improved due to CA-Jaccard using a Green border overlay and rank shift label.
- Format Gradio output galleries to render customized image captions showing rank indices, camera IDs, and optimization statuses.

## Capabilities

### New Capabilities
- `add-ui-visual-highlights`: Specification for border highlights, border color mappings, and result card text decoration.

### Modified Capabilities
<!-- None -->

## Impact

- Modifies search callback image rendering logic in [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py).
