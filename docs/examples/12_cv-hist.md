# Example [CV-HIST]: histograms, equalisation, CLAHE, normalisation

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-HIST] histograms, equalisation, CLAHE, normalisation
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
eq = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb); ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
eq_color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
bright = cv2.convertScaleAbs(img, alpha=1.3, beta=20)          # contrast, brightness
gamma = 0.5; lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
gamma_img = cv2.LUT(img, lut)
```
