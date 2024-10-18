
import pyzed.sl as sl
import cv2
from threading import Lock, Thread
from multiprocessing import Process
import numpy as np
from time import sleep
import logging
import sys,os, uuid
import numpy as np
from time import sleep
import redis
import json
sys.path.insert(0, os.getcwd()+'/flask_server/zed_artifacts/ultralytics/ultralytics') 
from ultralytics.models.yolo import YOLO
YOLO_TENSORRT_ENGINE = os.getcwd()+'/flask_server/bsu/artifacts/models/yolov8n.engine'
class ZEDDetector:
    def __init__(self, yolo_path=YOLO_TENSORRT_ENGINE) -> None:
        self.yolo_path = yolo_path
        self.exit_signal = False
        self.img_capture = None
        self.run_signal = False
        self.detection_start = False
        self.det = None
        self.detected_classes = None
        self.detections = None
        self.yolo = YOLO( self.yolo_path, task="detect")
        self.win_name = "Camera Remote Control"
        self.search_for_classes = ['person','tv']
        logging.root.setLevel(logging.NOTSET)
        logger = logging.getLogger(__name__)
        logging.basicConfig(format='%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.cam = None
        self.lock = Lock()
        self.redis_channel = "object_detection"
        self.redis_server = redis.Redis(host='localhost', port=6379, db=0)
        print(self.redis_server.publish("test","Hello"))

    def xywh2abcd(self, xywh, im_shape):
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

    def detections_to_custom_box(self,detections, im0,classes):
        output = []
        
        labels_list = []
        for i, det in enumerate(detections):
            xywh = det.xywh[0]

            # Creating ingestable objects for the ZED SDK
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = self.xywh2abcd(xywh, im0.shape)
            obj.label = det.cls[0]
            obj.probability = det.conf[0]
            obj.is_grounded = False
            obj.unique_object_id = sl.generate_unique_id()
            labels_list.append(classes[int(det.cls[0])])

            output.append(obj)
        return output,labels_list

    def detect_video(self, img_size, conf_thres=0.25,iou_thres=0.45):
        mat, runtime = self.startup_zed_camera()
        key = ''
        while not self.exit_signal:  # for 'q' key
            
            self.run_signal = True
            err = self.cam.grab(runtime) #Check that a new image is successfully acquired
            if err == sl.ERROR_CODE.SUCCESS:
                # self.lock.acquire()
                # while not self.exit_signal:
                self.cam.retrieve_image(mat, sl.VIEW.LEFT) #Retrieve left image
                # self.lock.release()
                cvImage = mat.get_data()
                img_capture = cvImage  
                #cv2.imshow(self.win_name, cvImage)                  
                if self.run_signal:
                    self.detection_start = True
                    self.lock.acquire()

                    img = cv2.cvtColor(img_capture, cv2.COLOR_BGRA2RGB)
                    predictions = self.yolo.predict( img,verbose=False,save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy()
                    capture_thread = Thread(target= self.process_detections, kwargs={'predictions': predictions, 'captured_image':img})
                    capture_thread.start()
                    self.lock.release()
                    self.detection_start = False
                    self.run_signal = False
                sleep(0.005)
                
            else:
                print("Error during capture : ", err)
                self.run_signal = False
                self.exit_signal = True
                break
            key = cv2.waitKey(5)
            if key == 113:
                self.detection_start = False
                self.run_signal = False
                self.exit_signal = True
        self.cam.close()        
    def stop(self):
        self.detection_start = False
        self.run_signal = False
        self.exit_signal = False
        self.cam.close()

    def process_detections(self, predictions=None, captured_image=None):
        logging.info("Starting ")
        if isinstance(captured_image, np.ndarray):
            det = predictions.boxes
            classes = predictions.names
            detections, labels = self.detections_to_custom_box(det, captured_image,classes)
            
            detected_classes = [classes[int(c)] for c in det.cls]
            
            indices = [i for i, elem in enumerate(detected_classes) if elem in self.search_for_classes]

            # if detection_start and detected_classes:
            if len(indices) > 0:
                for indx, elem in enumerate(detected_classes):
                    print(f'Detected class label: {detected_classes[indx]} index: {indx}')
                    person_box = det[indx].xyxy
                    person_box = list(person_box)[0]
                    x,y,w,h = person_box[0], person_box[1], person_box[2], person_box[3]
                    
                    xmin = int(x)

                    ymin = int(y)

                    xmax = int(w)

                    ymax = int(h) 
                    cropped_img = captured_image[ymin:ymax, xmin:xmax]
                    pred_conf = det[indx].conf
                    pred_class = det[indx].cls[0]
                    results = {
                        'pred_class': classes[pred_class],
                        'pred_prob': str(pred_conf[0])
                    }
                    print(results)
                    self.redis_server.publish(self.redis_channel, json.dumps(results))
                    #cv2.imwrite("image_"+str(uuid.uuid1())+".jpg", cropped_img)
    def startup_zed_camera(self):
        self.cam = sl.Camera()
        init_parameters = sl.InitParameters()
        init_parameters.depth_mode = sl.DEPTH_MODE.NONE
        init_parameters.sdk_verbose = 1

        
        
        status = self.cam.open(init_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(status)+". Exit program.")
            exit()
        runtime = sl.RuntimeParameters()
        
        mat = sl.Mat()
        #cv2.namedWindow(self.win_name)
        return mat, runtime
    def start(self):
        self.exit_signal = False
        capture_thread = Process(target= self.detect_video, kwargs={'img_size': 640})
        capture_thread.start()

        
        
        #cv2.destroyAllWindows()
        capture_thread.join()

    def stop(self):
        self.exit_signal = False

if __name__ == '__main__':
    zed = ZEDDetector('yolov8n.pt')
    zed.start()
