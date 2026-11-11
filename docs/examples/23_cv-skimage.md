# Example [CV-SKIMAGE]: scikit-image equivalents (float images in [0,1])

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-SKIMAGE] scikit-image equivalents (float images in [0,1])
from skimage import color, filters, feature, measure, morphology, segmentation, transform, util, exposure
im = util.img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
g = color.rgb2gray(im)
t = filters.threshold_otsu(g); bw = g > t
try:
    bw = morphology.remove_small_objects(bw, max_size=49)   # skimage >= 0.26
except TypeError:
    bw = morphology.remove_small_objects(bw, min_size=50)   # older skimage
lab = measure.label(bw)
for r in measure.regionprops(lab):
    area, (cy, cx), bbox, ecc = r.area, r.centroid, r.bbox, r.eccentricity
ed = feature.canny(g, sigma=2)
sob = filters.sobel(g)
rs = transform.resize(im, (150, 200), anti_aliasing=True)
rot = transform.rotate(im, 30)
eqa = exposure.equalize_adapthist(g)
slic = segmentation.slic(im, n_segments=50, start_label=1)
out8 = util.img_as_ubyte(eqa)                     # back to uint8 for cv2.imwrite
print("regions", lab.max())
```
