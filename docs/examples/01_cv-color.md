# Example [CV-COLOR]: colour spaces

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-COLOR] colour spaces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # for PIL / matplotlib
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          # H 0-179, S 0-255, V 0-255
lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)        # gray -> 3 channels (to draw in colour)
b, g, r = cv2.split(img); merged = cv2.merge([b, g, r])
```
