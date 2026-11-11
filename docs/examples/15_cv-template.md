# Example [CV-TEMPLATE]: template matching (single and multiple)

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-TEMPLATE] template matching (single and multiple)
templ = gray[60:140, 60:140]
res = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
th_, tw_ = templ.shape
cv2.rectangle(vis, max_loc, (max_loc[0] + tw_, max_loc[1] + th_), (0, 0, 255), 2)
ys, xs = np.where(res >= 0.8)                       # all matches above 0.8
print("best score", round(max_v, 3), "matches", len(xs))
```
