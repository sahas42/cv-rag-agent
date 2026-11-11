# Example [CV-NUMPY]: array idioms for images

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-NUMPY] array idioms for images
blank = np.zeros((h, w, 3), np.uint8); white = np.full((h, w), 255, np.uint8)
f32 = img.astype(np.float32) / 255.0; back8 = (f32 * 255).clip(0, 255).astype(np.uint8)
gray3 = np.stack([gray] * 3, axis=-1)
chw = np.transpose(img, (2, 0, 1)); hwc = np.transpose(chw, (1, 2, 0))   # for DL models
ys, xs = np.nonzero(otsu)                         # coordinates of white pixels
if len(xs): x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
img_copy = img.copy()                             # slices are views: copy before editing
img_copy[otsu > 0] = (0, 0, 255)                  # colour all mask pixels red
grid = np.hstack([img, img]); stack = np.vstack([grid, grid])
```
