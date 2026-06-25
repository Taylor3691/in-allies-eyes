## ADDED Requirements

### Requirement: Lazy loading of feature extractor
The system SHALL NOT load the PyTorch ResNet-50 feature extractor models during application start. The model initialization and checkpoint loading SHALL be deferred until a feature extraction operation is requested.

#### Scenario: Startup behavior
- **WHEN** the Gradio application is launched
- **THEN** PyTorch weights are not loaded, and the initialization time remains under one second.

### Requirement: Model loading on search or capture
The system SHALL automatically initialize the ResNet-50 feature extractor when the first query capture or gallery search is triggered.

#### Scenario: On-demand model initialization
- **WHEN** the user clicks "Capture Query" or "Search Gallery" for the first time
- **THEN** the system loads the ResNet-50 model, populates the weights from the pretrained checkpoint, and extracts the features for the query.

### Requirement: Memory caching of the model
The system SHALL cache the initialized model instance in memory after loading.

#### Scenario: Subsequent search request
- **WHEN** a subsequent capture or search request is executed
- **THEN** the system uses the cached model instance without reloading the weights from disk.
