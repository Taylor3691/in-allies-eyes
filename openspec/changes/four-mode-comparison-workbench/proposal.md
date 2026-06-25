## Why

Understanding the combined effect of Kalman tracking and CA-Jaccard re-ranking requires a side-by-side scientific comparison. A dedicated dashboard component displaying top-5 query match results across all four environment parameter combinations allows the evaluator to visually verify metric improvements.

## What Changes

- Implement a comparative search callback that takes both the raw detection crop and the Kalman-smoothed crop.
- Execute feature extraction and database query loops across four distinct modes: Mode 1 (Raw/Baseline), Mode 2 (Kalman/Baseline), Mode 3 (Raw/CAJ), and Mode 4 (Kalman/CAJ).
- Render results in four concurrent output galleries in the comparison tab.

## Capabilities

### New Capabilities
- `four-mode-comparison-workbench`: Specification of rendering and search parameters for the comparative dashboard.

### Modified Capabilities
<!-- None -->

## Impact

- Modifies [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py)'s comparison callback logic.
