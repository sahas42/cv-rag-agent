# Example [CV-HOUGH]: lines and circles

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-HOUGH] lines and circles
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=10)
if lines is not None:                               # shape (N,1,4) in 4.x, (N,4) in 5.x -> reshape
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
blur = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=30, minRadius=10, maxRadius=80)
circles = np.empty((0, 3)) if circles is None else circles.reshape(-1, 3)
for x, y, r in np.round(circles).astype(int):
    cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
print("lines:", 0 if lines is None else len(lines.reshape(-1, 4)), "circles:", len(circles))
```
