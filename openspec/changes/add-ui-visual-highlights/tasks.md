## 1. Border Highlight Overlay Implementation

- [x] 1.1 Add constant-value border drawing logic to search result images using `cv2.copyMakeBorder` in [search_gallery](file:///c:/Users/VICTUS/Documents/RecognitionCourse/in-allies-eyes/app.py#L112).
- [x] 1.2 Implement the border color mapping: Yellow `[255, 220, 0]` for same-camera match and Green `[46, 204, 113]` for rank-improved match.

## 2. Caption Text Customization

- [x] 2.1 Format image text captions to print rank index, camera ID, and camera bias tags under each baseline result item.
- [x] 2.2 Format Jaccard output captions to dynamically append rank improvement delta details (e.g. "Improved from X!").
