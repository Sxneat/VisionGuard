from ultralytics import YOLO

# 1. Load the YOLO model (can be pretrained like 'yolo11n.pt'
#    or your own weights, e.g. 'best.pt' from training)
model = YOLO("best.pt")   # or YOLO("best.pt")

# 2. Export to ONNX
model.export(format="onnx")   # creates 'yolo11n.onnx' in your working dir

# # 3. Load the exported ONNX model
# onnx_model = YOLO("yolo11n.onnx")

# # 4. Run inference
# results = onnx_model("https://ultralytics.com/images/bus.jpg")
# results.show()