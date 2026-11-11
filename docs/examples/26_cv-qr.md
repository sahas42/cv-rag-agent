# Example [CV-QR]: QR code detection (built in)

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-QR] QR code detection (built in)
data, pts, _ = cv2.QRCodeDetector().detectAndDecode(img)
print("QR:", data if data else "none")
```
