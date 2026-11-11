# Example [CV-ARITH]: blending, masks, bitwise

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-ARITH] blending, masks, bitwise
other = cv2.resize(cv2.flip(img, 1), (img.shape[1], img.shape[0]))   # same size required
blend = cv2.addWeighted(img, 0.7, other, 0.3, 0)
diff = cv2.absdiff(img, other)
sat = cv2.add(img, img)                       # saturates at 255 (img + img wraps around!)
mask = np.zeros(img.shape[:2], np.uint8); cv2.circle(mask, (w // 2, h // 2), 80, 255, -1)
roi = cv2.bitwise_and(img, img, mask=mask)
inv_mask = cv2.bitwise_not(mask)
```
