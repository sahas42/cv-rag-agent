# Example [CV-PERSPECTIVE]: warp a quadrilateral to a rectangle

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-PERSPECTIVE] warp a quadrilateral to a rectangle
src = np.float32([[50, 40], [350, 60], [370, 260], [30, 280]])   # TL, TR, BR, BL
dst = np.float32([[0, 0], [300, 0], [300, 200], [0, 200]])
P = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, P, (300, 200))
```
