# Example [CV-CONTOURS]: find, filter, measure, draw (OpenCV 4/5: two return values)

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-CONTOURS] find, filter, measure, draw (OpenCV 4/5: two return values)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) > 100]          # drop noise
contours = sorted(contours, key=cv2.contourArea, reverse=True)
vis = img.copy()
for c in contours:
    area = cv2.contourArea(c); perim = cv2.arcLength(c, True)
    x, y, bw, bh = cv2.boundingRect(c)
    m = cv2.moments(c)
    cx, cy = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) if m["m00"] else (x, y)
    approx = cv2.approxPolyDP(c, 0.02 * perim, True)              # len(approx)==4 -> quadrilateral
    circularity = 4 * np.pi * area / (perim * perim) if perim else 0   # 1.0 = perfect circle
    (ex, ey), radius = cv2.minEnclosingCircle(c)
    rect = cv2.minAreaRect(c); box = cv2.boxPoints(rect).astype(np.intp)
    hull = cv2.convexHull(c)
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    cv2.drawContours(vis, [box], 0, (0, 0, 255), 1)
    cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)
cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)              # -1 = all
print("objects:", len(contours))
cv2.imwrite("contours.png", vis)
```
