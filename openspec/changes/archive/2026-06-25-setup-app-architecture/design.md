## Context

The Person Re-ID demo application requires a Gradio-based graphical interface and a Kalman Filter tracking component. In this initial stage, we are setting up the structure and placeholders for these features to establish a clear architectural pattern without adding code complexity.

## Goals / Non-Goals

**Goals:**
- Initialize the entry point [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py) with the 3-tab layout outlined in the FSD.
- Scaffold the Kalman tracker module [tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py) with standard tracking stubs (`predict` and `update` methods).
- Ensure the app launches successfully via `python app.py` with placeholder components.

**Non-Goals:**
- Implementing actual computer vision models (face/person detection).
- Coding the mathematical equations of the Kalman Filter.
- Running live distance matrix re-ranking or database loading.

## Decisions

### Decision 1: Gradio Blocks Layout
We choose Gradio Blocks API in [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py) over Interface API because we require deep customization (columns, sidebars, double-row grids) to present the comparative analysis grids side-by-side.

### Decision 2: Tracker Class Object-Oriented Structure
We define a clean class interface in [tracker.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/src/Kalman_filter/tracker.py):
```python
class KalmanTracker:
    def __init__(self):
        self.state = None
        self.status = "Missing" # "Found" or "Missing"

    def predict(self):
        pass

    def update(self, bbox):
        pass
```
This isolates the tracking math from the Gradio UI thread.

## Risks / Trade-offs

* **[Risk]** The Gradio app might crash if it tries to load model weights or read video streams without real cameras.
  * **Mitigation:** Use Gradio's standard dummy placeholder components and load weights lazily inside event functions rather than globally at script execution time.
