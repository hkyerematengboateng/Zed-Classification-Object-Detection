'''
Service that performs inference and confidence score computations
and results back to the controller.
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from .ls_stat import LS_Stat
MODEL_PATH = "/home/user1/ros2_models/models/acoustic_silence_mel_32_ds_lrelu_v2.tflite"
#MODEL_PATH = "models/acoustic_silence_mel_32_ds_prelu_v1.tflite"
ACTIVATION_STATISTICS_PATH = ""

class Model_Service(object):
    def __init__(self, statistics, image_file, image_shape: list, model_interpreter=None) -> None:
        
        self.image_task = image_file
        self.image_input_shape = image_shape
        if model_interpreter is None:
            self.model_path = MODEL_PATH
            self.model_interpreter = self.load_tflife_model()
        else:
            self.model_interpreter = model_interpreter
        self.ls_service = LS_Stat(self.model_interpreter,"monitorting_node", self.image_task )
        self.statistics = statistics
        self.class_labels = [1,2,3,4,5,6,7,8,9]


    def load_model(self):
        return self.load_tflife_model()
    def load_statisitcs(self):
        return pd.read_pickle(self.statistics_path)
    
    def perform_analysis(self, test_data, true_class):
        if self.image_task:
            test_data_reshape = np.reshape(test_data,(self.image_input_shape[0], self.image_input_shape[1]))
        else:
            test_data_reshape = np.reshape(test_data,(-1,1))
        y_pred = self.__perform_evaluation(test_data_reshape)
        
        test_activations = self.__get_data_activation(test_data_reshape)
        predicited_class = np.argmax(y_pred, axis=0)
        predicted_prob = y_pred[predicited_class]
    
        predicited_class_label = predicited_class + 1
        print('Predicted class is {}'.format(predicited_class_label))
        results = self.ls_service.process(test_activations, predicited_class_label,self.statistics)
        results['Prediction Probability'] = predicted_prob
        #results['True Class'] = true_class
        return results.to_json(orient='records')
    
    def __get_data_activation(self,test_ds):
        model_interpreter = self.model_interpreter
        if model_interpreter is not None:
            input_details = model_interpreter.get_input_details()
            
            model_interpreter.set_tensor(input_details[0]['index'], [test_ds])
            return model_interpreter.get_tensor(4)
        else:
            return []
        
    def load_tflife_model(self):
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter

    def __perform_evaluation(self,test_data):
        interpreter= self.model_interpreter

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], [test_data])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data[0]
    