import numpy as np
import pickle, os
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

import tensorflow as tf
import keras_cv
from ultralytics import YOLO
import cv2
from keras import layers
import keras

stackwise_channels = [128, 256, 512, 1024]
stackwise_depth = [3, 9, 9, 3]
include_rescaling = False

yolo_backbone = keras_cv.models.YOLOV8Backbone(
    stackwise_channels,
    stackwise_depth,
    include_rescaling,
    activation="swish",
    input_shape=(None, None, 3),
    input_tensor=None
    )
IMAGENET_DATASET_PATH = os.getcwd()+"/datasets/n01443537"
img = IMAGENET_DATASET_PATH+"/n01443537_16.JPEG"

img_obj = cv2.imread(img)
img_obj = img_obj[tf.newaxis,...]
output = yolo_backbone(img_obj)
print(f'Output shape {output.shape}')
#yolo_backbone.summary()

model = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_xs_backbone_coco"
)
model.trainable = False
base_model = model
print(f'Image shape {img_obj.shape}')

inputs = keras.Input(shape=(None,None,3))

x = base_model(inputs,training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation='relu')(x)
outputs = keras.layers.Dense(20,activation='softmax')(x)

final_model = keras.Model(inputs, outputs)
# output_bb = model.predict(img_obj)
# output_bb = keras_cv.bounding_box.to_dense(output_bb, max_boxes=None, default_value=-1)
print(f'{final_model.summary()}')
# keras.models.load_model('/home/user1/Downloads/VGGnet_fast_rcnn_iter_70000.ckpt')
# keras.layers.TFSMLayer('/home/user1/Downloads/VGGnet_fast_rcnn_iter_70000.ckpt', call_endpoint='serving_default')