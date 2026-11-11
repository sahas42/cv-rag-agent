# Example [CV-BLUR]: smoothing / denoising (kernel sizes odd and positive)

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-BLUR] smoothing / denoising (kernel sizes odd and positive)
g5   = cv2.GaussianBlur(img, (5, 5), 0)
med  = cv2.medianBlur(img, 5)                        # salt-and-pepper noise
bil  = cv2.bilateralFilter(img, 9, 75, 75)           # edge-preserving
box  = cv2.blur(img, (3, 3))
nlm  = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
sharp = cv2.filter2D(img, -1, k)                     # custom kernel (sharpen)
```
