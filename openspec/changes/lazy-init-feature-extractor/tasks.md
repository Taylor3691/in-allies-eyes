## 1. Feature Extractor Model Lazy Loading

- [x] 1.1 Implement a thread-safe helper function in [app.py](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py) to load the ResNet-50 PyTorch model checkpoint.
- [x] 1.2 Add logic to verify if the model is already initialized, caching it in a global variable for subsequent requests.

## 2. Callback Integration and UI Progress Indicators

- [x] 2.1 Integrate the model lazy-loader into [capture_query](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py#L135).
- [x] 2.2 Integrate the model lazy-loader into [search_gallery](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py#L26) and ensure the loading indicator is displayed.
