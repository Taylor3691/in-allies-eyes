## 1. Comparison Callback Implementation

- [x] 1.1 Implement `run_comparison` callback inside [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py) to run query searches sequentially.
- [x] 1.2 Setup the four modes using correct parameters: raw crop with cosine (Mode 1), Kalman crop with cosine (Mode 2), raw crop with CAJ (Mode 3), and Kalman crop with CAJ (Mode 4).

## 2. Event Handler Wiring

- [x] 2.1 Wire the `Run 4-Mode Comparison` button event handler to trigger `run_comparison` callback in Gradio UI.
- [x] 2.2 Add Gradio loading indicators and progress states on all 4 output galleries concurrently.
