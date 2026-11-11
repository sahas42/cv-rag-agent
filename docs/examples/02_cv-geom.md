# Example [CV-GEOM]: resize, crop, flip, rotate, pad

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-GEOM] resize, crop, flip, rotate, pad
small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)    # dsize = (width, height)
big   = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
crop  = img[50:150, 80:200]                          # [y1:y2, x1:x2]
flip_h = cv2.flip(img, 1); flip_v = cv2.flip(img, 0)
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)  # centre, angle (deg, CCW), scale
rot = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))
T = np.float32([[1, 0, 25], [0, 1, 10]]); shifted = cv2.warpAffine(img, T, (w, h))
padded = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
```
