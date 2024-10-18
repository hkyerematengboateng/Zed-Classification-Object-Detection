import numpy as np
import pickle, os
from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn

def add_fc_layer(model, fc_size):
    # Get the current model structure
    backbone = model.model.model[:-1]  # all layers except the last one
    current_head = model.model.model[-1]  # the last layer (classification head)
    out_features = current_head.linear.out_features
    print(out_features)
    # Create a new sequential module for the head
    # Create a new sequential module for the additional layers
    additional_layers = nn.Sequential(
        nn.Linear(out_features, fc_size),
        nn.ReLU(),
        nn.Linear(fc_size, out_features)
    )
    
    # Create a new Classify module
    new_head = nn.Sequential(
        current_head.conv,
        current_head.pool,
        current_head.drop,
        current_head.linear,
        additional_layers
    )
    
    # Replace the old classification head with the new one
    model.model.model[-1] = new_head
    
    
    return model

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

yolo = YOLO('yolov8n-cls.pt')

# Add a new fully connected layer with 256 units
fc_size = 256
modified_model = add_fc_layer(yolo, fc_size)
print(modified_model)
# Save the modified model
modified_model.save('yolov8n-cls-modified.pt')
# yolo.add_callback('train_backbone', freeze_layer)
IMAGENET_DATASET_PATH = os.getcwd()+"/datasets/n01443537"
# activations = {}
# hooks = {}
# def get_activation(name):
#     def hook(yolo, input, output):
#         activations[name] = output.detach().cpu().numpy()
#     return hook

# node = "conv"
# yolo.model.model[9].register_forward_hook(get_activation(node))
img = IMAGENET_DATASET_PATH+"/n01443537_16.JPEG"

img_obj = cv2.imread(img)

yolo_mod = YOLO('yolov8n-cls-modified.pt')
preds = yolo_mod(img_obj)
print(preds)
# features = activations[node]

# print(features.shape)

