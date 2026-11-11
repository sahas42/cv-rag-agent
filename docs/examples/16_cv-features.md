# Example [CV-FEATURES]: keypoints, matching, homography

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-FEATURES] keypoints, matching, homography
img2 = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0), (w, h))
g1, g2 = gray, cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=1000)
k1, d1 = orb.detectAndCompute(g1, None); k2, d2 = orb.detectAndCompute(g2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(d1, d2), key=lambda m: m.distance) if d1 is not None and d2 is not None else []
match_vis = cv2.drawMatches(img, k1, img2, k2, matches[:30], None, flags=2)
sift = cv2.SIFT_create()                            # in main module since OpenCV 4.4
s1, sd1 = sift.detectAndCompute(g1, None); s2, sd2 = sift.detectAndCompute(g2, None)
good = []
if sd1 is not None and sd2 is not None and len(s2) >= 2:
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(sd1, sd2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]  # Lowe ratio
if len(good) >= 4:
    src = np.float32([s1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([s2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(img, H, (w, h))
print("orb matches", len(matches), "sift good", len(good))
```
