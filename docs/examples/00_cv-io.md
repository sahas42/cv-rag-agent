# Example [CV-IO]: read / write / info

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-IO] read / write / info
import cv2, numpy as np, os
img = cv2.imread("input.jpg")                       # BGR uint8, None if missing
if img is None:
    img = np.zeros((240, 320, 3), np.uint8)          # synthetic fallback
gray = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
rgba = cv2.imread("input.jpg", cv2.IMREAD_UNCHANGED)  # keeps alpha / 16-bit
h, w = img.shape[:2]; ch = img.shape[2] if img.ndim == 3 else 1
print(h, w, ch, img.dtype, img.min(), img.max())
cv2.imwrite("out.png", img)
cv2.imwrite("out.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
os.makedirs("results", exist_ok=True)               # imwrite returns False if folder missing
cv2.imwrite(os.path.join("results", "a.png"), img)
```
