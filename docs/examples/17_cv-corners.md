# Example [CV-CORNERS]: 

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-CORNERS]
harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
corner_mask = harris > 0.01 * harris.max()
pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
n_corners = 0 if pts is None else len(pts)
```
