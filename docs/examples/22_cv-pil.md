# Example [CV-PIL]: PIL <-> OpenCV <-> matplotlib

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-PIL] PIL <-> OpenCV <-> matplotlib
from PIL import Image, ImageFilter, ImageOps
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))       # OpenCV -> PIL
back = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)             # PIL -> OpenCV
pil.convert("L").filter(ImageFilter.GaussianBlur(2)).save("pil_blur.png")
ImageOps.autocontrast(pil).resize((200, 150)).save("pil_small.png")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title("input"); ax[0].axis("off")
ax[1].plot(hist); ax[1].set_title("histogram")
fig.savefig("plot.png", dpi=100, bbox_inches="tight"); plt.close(fig)
```
