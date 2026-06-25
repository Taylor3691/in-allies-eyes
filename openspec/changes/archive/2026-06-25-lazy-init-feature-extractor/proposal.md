## Why

The Re-ID demo application requires loading heavy deep learning models (PyTorch ResNet-50) which causes a startup delay of several seconds if initialized at launch. To ensure the application starts up in under one second, the model loading must be deferred and initialized lazily on-demand when the first search or query capture is triggered.

## What Changes

- Add a lazy loader module or function to dynamically instantiate and load the ResNet-50 model weights from the pretrained checkpoint.
- Update the webcam query capture and search functions to trigger model initialization on the first invoke.
- Cache the loaded PyTorch model in a global variable to avoid repeated disk reads and initialization overhead for subsequent operations.

## Capabilities

### New Capabilities
- `lazy-init-feature-extractor`: Specification of requirement rules for deferred model weight loading and memory caching.

### Modified Capabilities
<!-- None -->

## Impact

- Modifies the initialization and callback sequences in [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py).
- Interacts with pretrained weights in the `pretrained_models/` folder.
