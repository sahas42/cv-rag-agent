# Example [CV-VIDEO]: write and read a video; background subtraction; optical flow

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-VIDEO] write and read a video; background subtraction; optical flow
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter("demo.mp4", fourcc, 10, (w, h))      # size = (width, height)
for i in range(20):
    frame = np.zeros((h, w, 3), np.uint8); cv2.circle(frame, (20 + 15 * i, h // 2), 15, (0, 255, 0), -1)
    vw.write(frame)
vw.release()
cap = cv2.VideoCapture("demo.mp4")                  # 0 = webcam (not available headless)
fps = cap.get(cv2.CAP_PROP_FPS); nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
bg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)
prev = None; n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fgmask = bg.apply(frame)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev is not None:
        flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    prev = g; n += 1
cap.release()
print("frames read", n, "fps", fps)
```
