import os
# from ultralytics import YOLO
from pycocotools.coco import COCO
import cv2
import pandas as pd
import sys
from time import sleep
sys.path.insert(0, os.getcwd()+'/ultralytics/ultralytics') 
from models.yolo import YOLO
# model = YOLO('runs/detect/train/weights/best.pt') # load a detect pre-trained model 
# model = YOLO('yolov8n.pt')
# model_cls = YOLO('yolov8n-cls.pt')
# Export the model

# model.export(format="engine",dynamic=True,simplify=True,half=True)  # creates 'yolov8n.engine'
# model_cls.export(format="engine",dynamic=True,simplify=True,half=True)

# Create a TensorRT runtime object
trt_model = YOLO('yolov8n-cls.pt')
print(trt_model)
# Run inference
# results = trt_model("bus.jpg")
# print(results)
# sleep(5)