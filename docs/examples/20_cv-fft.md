# Example [CV-FFT]: frequency domain low-pass filter

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-FFT] frequency domain low-pass filter
f = np.fft.fftshift(np.fft.fft2(gray))
mag_spec = 20 * np.log(np.abs(f) + 1)
cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
lp = np.zeros_like(gray, np.float32); cv2.circle(lp, (cx, cy), 30, 1, -1)
low = np.abs(np.fft.ifft2(np.fft.ifftshift(f * lp)))
low = cv2.normalize(low, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
```
