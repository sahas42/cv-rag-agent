# Example [CV-MORPH]: morphology on a binary mask

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-MORPH] morphology on a binary mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # or np.ones((5,5), np.uint8)
eroded  = cv2.erode(otsu, kernel, iterations=1)
dilated = cv2.dilate(otsu, kernel, iterations=1)
opened  = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)    # remove small specks
closed  = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)   # fill small holes
grad    = cv2.morphologyEx(otsu, cv2.MORPH_GRADIENT, kernel)
tophat  = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
```
