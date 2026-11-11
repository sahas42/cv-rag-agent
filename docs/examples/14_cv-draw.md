# Example [CV-DRAW]: annotate (coordinates are (x, y) ints; colours BGR)

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-DRAW] annotate (coordinates are (x, y) ints; colours BGR)
out = img.copy()
cv2.line(out, (0, 0), (w - 1, h - 1), (255, 0, 0), 2)
cv2.rectangle(out, (10, 10), (110, 60), (0, 255, 0), 2)          # -1 thickness = filled
cv2.circle(out, (w // 2, h // 2), 30, (0, 0, 255), -1)
cv2.ellipse(out, (200, 100), (60, 30), 0, 0, 360, (255, 255, 0), 2)
pts = np.array([[50, 200], [100, 150], [150, 220]], np.int32)
cv2.polylines(out, [pts], True, (0, 255, 255), 2); cv2.fillPoly(out, [pts], (80, 80, 80))
cv2.putText(out, "count: 3", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
cv2.arrowedLine(out, (300, 50), (360, 90), (0, 0, 255), 2)
```
