# Example [CV-NOISE]: add noise / quality metrics

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-NOISE] add noise / quality metrics
gauss = np.clip(img.astype(np.float32) + np.random.normal(0, 15, img.shape), 0, 255).astype(np.uint8)
sp = img.copy(); r = np.random.rand(*img.shape[:2]); sp[r < 0.02] = 0; sp[r > 0.98] = 255
psnr = cv2.PSNR(img, gauss)
mse = np.mean((img.astype(np.float64) - gauss) ** 2)
from skimage.metrics import structural_similarity as ssim
s = ssim(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY), data_range=255)
print("PSNR", round(psnr, 2), "SSIM", round(s, 3))
```
