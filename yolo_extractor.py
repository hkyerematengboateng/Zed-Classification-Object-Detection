import torch
import torch.nn as nn
from torchvision import models, datasets
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import os



class YOLOFeatureExtractor(nn.Module):
    def __init__(self, yolo_model='yolov8n.pt'):
        super(YOLOFeatureExtractor, self).__init__()
        self.yolo = YOLO(yolo_model)
        # Get the backbone and neck of YOLO
        self.xx = self.yolo.model.model[-1]
        
        self.feature_extractor = nn.Sequential(*list(self.yolo.model.model.children())[:-1])

    def forward(self, x):
        return self.feature_extractor(x)

class YOLOResNetClassifier(nn.Module):
    def __init__(self, num_classes, yolo_model='yolov8n-cls.pt'):
        super(YOLOResNetClassifier, self).__init__()
        
        self.yolo_extractor = YOLOFeatureExtractor(yolo_model)
        
        # Freeze YOLO parameters
        for param in self.yolo_extractor.parameters():
            param.requires_grad = False
        
        # Load a pre-trained ResNet model
        self.resnet = models.resnet50()
        
        # Replace the first conv layer to match YOLO output channels
        self.resnet.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        # self.resnet.fc1 = nn.Linear(num_ftrs, 512)
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        print(f'Input shape: {x.shape}')
        # Extract features using YOLO
        yolo_features = self.yolo_extractor(x)
        x = self.resnet(x)  # extract features using ResNet

        x = x.view(-1, 2048)  # flatten the output

        x = self.fc(x)  # add a fully connected layer
        # Pass YOLO features through ResNet
        return self.resnet(yolo_features)

def preprocess_image(image_path):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Example usage
if __name__ == "__main__":
    # Initialize the model
    num_classes = 10  # Adjust based on your classification task
    model = YOLO('yolov8n-cls.pt')

    IMAGENET_DATASET_PATH = os.getcwd()+"/datasets/n01443537"
    # Prepare an example image
    image_path = IMAGENET_DATASET_PATH+"/n01443537_318.JPEG"

    # img_obj = cv2.imread(img)

    # image_path = "path/to/your/image.jpg"
    input_tensor = preprocess_image(image_path)
    # Load a pre-trained ResNet model
    resnet = models.resnet101()
    # Debug: Print shapes at each step
    print(f"Input tensor shape: {input_tensor.shape}")
    
    output = resnet(input_tensor)
    print(f"Final output shape: {output.shape}")
    
    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    print(f"Predicted class: {predicted_class.item()} ")

    preds = model(input_tensor)