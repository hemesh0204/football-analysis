from ultralytics import YOLO
import torch

# model = YOLO("yolov8x") we were using this model at the begining, and we will be using the model's weight for which we have trained on those augmentated images.

model = YOLO("models/best.pt")

result = model.predict("input/08fd33_4.mp4", save=True)
# first Frame
print(result[0])
print("--------------------------------------------------------------")
for box in result[0].boxes:
    print(box)
