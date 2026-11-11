# Example [CV-DNN]: run a local ONNX model with OpenCV (no internet; file must exist)

Tested OpenCV/NumPy/scikit-image example. Variables such as img, gray, otsu, edges, vis, kernel, h, w come from earlier examples (00_cv-io: img = cv2.imread(...), h, w = img.shape[:2]).

```python
# [CV-DNN] run a local ONNX model with OpenCV (no internet; file must exist)
model_path = "model.onnx"
if os.path.exists(model_path):
    net = cv2.dnn.readNetFromONNX(model_path)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, size=(224, 224), swapRB=True, crop=False)
    net.setInput(blob); out = net.forward()
    print("pred", int(np.argmax(out)))
```
