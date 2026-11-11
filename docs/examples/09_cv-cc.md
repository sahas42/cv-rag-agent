# Example [CV-CC]: connected components with stats (fast object counting)

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-CC] connected components with stats (fast object counting)
n, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu, connectivity=8)
objs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 100]   # label 0 = background
print("count:", len(objs))
for i in objs:
    x, y, bw, bh, area = stats[i]
colored = cv2.applyColorMap((labels * (255 // max(n - 1, 1))).astype(np.uint8), cv2.COLORMAP_JET)
```
