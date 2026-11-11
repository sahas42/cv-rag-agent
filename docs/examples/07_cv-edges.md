# Example [CV-EDGES]: edges and gradients

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-EDGES] edges and gradients
edges = cv2.Canny(gray, 50, 150)                     # input uint8 gray; thresholds low, high
v = np.median(gray); auto = cv2.Canny(gray, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
mag = cv2.convertScaleAbs(cv2.magnitude(sx, sy))     # back to uint8 for saving
lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
cv2.imwrite("edges.png", edges)
```
