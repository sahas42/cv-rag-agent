# Example [CV-SEGMENT]: k-means colour quantisation, watershed, GrabCut

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-SEGMENT] k-means colour quantisation, watershed, GrabCut
Z = img.reshape(-1, 3).astype(np.float32)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, lbl, centers = cv2.kmeans(Z, 4, None, crit, 3, cv2.KMEANS_PP_CENTERS)
quant = centers.astype(np.uint8)[lbl.ravel()].reshape(img.shape)

# watershed: separate touching objects
dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0); sure_fg = sure_fg.astype(np.uint8)
sure_bg = cv2.dilate(opened, kernel, iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg); markers = markers + 1; markers[unknown == 255] = 0
markers = cv2.watershed(img.copy(), markers)        # boundaries = -1
n_regions = len(np.unique(markers)) - 2

# GrabCut from a rectangle
gmask = np.zeros(img.shape[:2], np.uint8); bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
cv2.grabCut(img, gmask, (50, 50, w - 100, h - 100), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
fg = np.where((gmask == cv2.GC_FGD) | (gmask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
```
