import os
from ultralytics import YOLO
from pycocotools.coco import COCO
import cv2
import pandas as pd
from ultralytics.utils.plotting import Annotator
# model = YOLO('runs/detect/train/weights/best.pt') # load a detect pre-trained model 
model = YOLO('yolov8n.pt')
model_cls = YOLO('yolov8n-cls.pt')
coco_path =  os.getcwd()+"/pytorch_yolov8/datasets/coco/"
coco_annotations_val_path = coco_path+"annotations/instances_val2017.json"
coco_annotations_train_path = coco_path+"annotations/instances_train2017.json"
coco = COCO(coco_annotations_val_path)
# g = coco.imgs
# # print(g)
# r = coco.cats
labels = list( coco.cats.keys())

# catIds = coco.getCatIds(catIds=labels)

# imgIds = coco.getImgIds(catIds=[2])
# images = coco.loadImgs(imgIds)

print('-'*19)
val_pd = None
val_dataset = []
count = 0
for label in labels:
    imgLbls = coco.getImgIds(catIds=[label])

    img_id = coco.loadImgs(imgLbls[0])
    filename = label, img_id[0]['file_name']
    
    img_file_path = coco_path+"images/val2017/"+filename[1]
    img = cv2.imread(img_file_path)
    orig_img = img
    val_dataset.append([img, label])
    
    count +=1
    pred = model(img, save=True,save_crop=True)
    img_boxes = pred[0].boxes
    class_id = img_boxes[0].cls.cpu().tolist()[0]
    names= pred[0].names
    print(f"Prediction probability: {img_boxes[0].conf}")
    class_name = names[class_id]
    print(f"Class Label: {class_name}")
    person_box = img_boxes[0].xyxy.cpu().tolist()[0]
    #person_box = list(person_box)

    x,y,w,h = person_box[0], person_box[1], person_box[2], person_box[3]
    print(f'x= {x}')
    xmin = int(x)

    ymin = int(y)

    xmax = int(w)

    ymax = int(h) 
    cropped_img = orig_img[ymin:ymax, xmin:xmax]
    pred_cls = model_cls(cropped_img, save=True)
    print(f"Classification result: {pred_cls}")
    #print(pred[0].pred)
    cv2.imwrite('original_image.jpg',orig_img)
    cv2.imwrite('detected_image.jpg',cropped_img)
    if count > 0:
    
        
        break
# val_pd = pd.DataFrame(val_dataset)
# print(f'Size of validation image {len(val_dataset)} {val_pd.shape}')

# results = model.train(data="coco.yaml", epochs=2, imgsz=640)
# Export the model

# model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
# trt_model = YOLO("runs/detect/train/weights/best.engine")
# print(model_cls)