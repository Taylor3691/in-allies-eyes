## ADDED Requirements

### Requirement: App Scaffold Layout
The system SHALL define a Gradio Blocks layout containing three main tabs: "Realtime Capture / Kalman Demo", "Gallery Search / CA-Jaccard Demo", and "Four-Mode Top-5 Comparison".

#### Scenario: Verify interface layout is present
- **WHEN** the user runs `python app.py`
- **THEN** the Gradio web server launches successfully and displays the three specified tabs.

### Requirement: Lazy Weight Loading
The system SHALL defer PyTorch model weight initialization and gallery embeddings cache loading until a feature extraction query is actively requested by the user.

#### Scenario: Validate zero model startup latency
- **WHEN** the application starts up
- **THEN** no deep learning models are loaded into memory and the startup sequence completes in under 2 seconds.
