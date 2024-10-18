import numpy as np
import pandas as pd
import statistics
import tensorflow as tf

def generate_stat_props(class_dataset, class_labels):
    if class_dataset:
        tano = Tano(training_dataset=class_dataset)
        tano.generate_train_distributions(train_labels=class_labels)

    return tano.dists
class Tano:
    def __init__(self, model,training_dataset) -> None:
        dists, classes, class_ids = dict(), dict, dict
        self.model =  model
        self.dists = dists
        self.classes = classes
        self.class_ids = class_ids
        self.training_dataset = training_dataset

    def generate_train_distributions(self, train_labels):
        self.training_dataset

        train_activations = self.get_monitoring_node_output(self.model,self.training_dataset)
            
        train_df = pd.DataFrame(train_activations)

        train_df['label'] = train_labels
        train_distribution_list = []

        for  i,label in enumerate(self.classes):

            class_data = train_df.loc[train_df.label == label]
            
            class_data = class_data.drop(columns=['label'])
            distribution_results = self.method_stats(class_data.values.transpose(),"triangle")
            dist_re = {'ClassLabel':label,"Distributions":distribution_results}
            train_distribution_list.append(dist_re)
        return train_distribution_list

    def  method_stats(self, dist_data, dist_names):
        distribution = []
        for i in dist_data:
            props = self.calculate_dis_props(i, distribution, dist_names)
            distribution.append(props)
        return distribution
    def calculate_dis_props(self, dist_data, distribution, dist_names):
        dist_name =   dist_names
        dist_data = dist_data #+ small_val
        max_val = np.amax(dist_data)
        min_val = np.amin(dist_data)
        peak_val = statistics.mode(dist_data)
        distribution = {'Type of Distribution':dist_name,'Mean': np.mean(dist_data),'StandardDeviation': np.std(dist_data),'Median':np.median(dist_data),'Min' : min_val,'Max' :max_val,'Peak':peak_val}
        return distribution
    
    def __get_monitoring_node_output(self):
        self.model.trainable = False
        monitoring_node = self.get_monitoring_node(self.model, self.monitoring_node)
    
        test_monitored_output = monitoring_node(self.dataset)
        output = test_monitored_output.numpy()
        return output
    
    
    def __get_monitoring_node(self,node:str):
        self.model.trainable = False
        monitoring_node = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer(name=node).output,
        )
        return monitoring_node 