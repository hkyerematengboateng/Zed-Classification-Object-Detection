import cv2
from ultralytics import YOLO
from threading import Lock, Thread
import numpy as np
from time import sleep
import logging
logger = logging.getLogger(__name__)
import requests
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Open the default camera (index 0)
run_signal = False
exit_signal = False
img_capture = None
lock = Lock()
cap = None
def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

def detections_to_custom_box(det, im0):
    output = []
    detections = det.boxes
    classes = det.names
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        logging.info('Classes',classes[det.cls[0]])
    return det
def detect_video(weights, img_size, conf_thres=0.09,iou_thres=0.45):
    global run_signal, exit_signal,img_capture,cap
    logging.info('Starting')
    yolo = YOLO(weights)
    while not exit_signal:
        if run_signal:
            lock.acquire()
            img = cv2.cvtColor(img_capture, cv2.COLOR_BGRA2RGB)
            predictions = yolo.predict( img,save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy()
            logging.info(predictions)
            det = predictions.boxes
            detections = detections_to_custom_box(predictions, img_capture)
            
            lock.release()
        sleep(0.5)
def load_video():
    global cap, img_capture,run_signal, exit_signal
    capture_thread = Thread(target=detect_video, kwargs={'weights': 'yolov8n.pt', 'img_size': 640})
    capture_thread.start()
    
    cap = cv2.VideoCapture(0)
    # Check if the camera is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    # 
    while not exit_signal:
        lock.acquire()
        # Capture frame-by-frame
        ret, frame = cap.read()
        # lock.release()
        # Check if a frame was returned
        if not ret:
            print("Cannot receive frame")
            break

        # lock.acquire()
        # Display the frame
        cv2.imshow('frame', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('c'):
            img_capture = frame
            run_signal = True
            logger.info('Photo captured')
            sleep(0.5)

            run_signal = False
        if cv2.waitKey(1) == ord('q'):
            print('Stoping camera')
            run_signal = False
            exit_signal = False
            break
            # break
        lock.release()
    cap.release()
    cv2.destroyAllWindows()

# Release the camera and close the window



if __name__ == '__main__':
    load_video()
