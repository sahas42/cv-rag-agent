# Example [CV-HSV-MASK]: segment a colour range

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-HSV-MASK] segment a colour range
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))      # red wraps around H=0/180
red2 = cv2.inRange(hsv, (170, 100, 100), (179, 255, 255))
red = cv2.bitwise_or(red1, red2)
green = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))
blue = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
only_red = cv2.bitwise_and(img, img, mask=red)
print("red pixels:", cv2.countNonZero(red))
```
