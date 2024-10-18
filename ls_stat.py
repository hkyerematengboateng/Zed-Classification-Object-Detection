#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:34:39 2023

@author: hkyeremateng-boateng

LS-Stat Algorithm Service
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import statistics
# import brier_score

class LS_Stat:
    def __init__(self, model, node_name, image_file=False):
        print("Starting......")
        self.image_task = image_file
        # self.tcsa_model = model
        self.monitoring_node = node_name
        self.other_methods = True
    
    def generate_train_distributions(self, train_dataset, train_labels, classes,loaded_train_distributions=None):
        if loaded_train_distributions is None:
            train_activations = self.get_monitoring_node_output(self.tcsa_model,train_dataset)
        else:
            train_activations = loaded_train_distributions
            
        train_df = pd.DataFrame(train_activations)

        train_df['label'] = train_labels
        train_distribution_list = []
        print(train_df.shape)
        for  i,label in enumerate(classes):
            print(label)
            class_data = train_df.loc[train_df.label == label]
            
            class_data = class_data.drop(columns=['label'])
            distribution_results = self.method_stats(class_data.values.transpose(),"triangle")
            dist_re = {'ClassLabel':label,"Distributions":distribution_results}
            train_distribution_list.append(dist_re)
        return train_distribution_list

    def __predict_softmax(self,loaded_model,test_ds):
        y_pred = loaded_model.predict(test_ds)
        pred = np.argmax(y_pred,axis=1)
        y_max = tf.math.reduce_max(y_pred,axis=1).numpy()
        return pred, y_max
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
     
    def _calculate_confidence_score(self,class_distribution_props, ds_dataset):
        activations = np.reshape(ds_dataset,(-1,1))
        # print('Num of features {} activations {} shape {}'.format(numoffeatures, activations.shape,test_data.shape))
        total_cost = 0
        total_guassian_1 = 0
        total_guassian_2 = 0
        total_guassian_3 = 0
        total_median = 0
        for index,distribution in enumerate(class_distribution_props):
            sumanomaly = 0
            sumanomaly = self._computeAnomaly(distribution, activations[index])
            total_cost +=sumanomaly
            
            guassian_cost_1 = self._calculate_gaussian(distribution, activations[index],1)
            total_guassian_1 +=guassian_cost_1
            
            guassian_cost_2 = self._calculate_gaussian(distribution, activations[index],2)
            total_guassian_2 +=guassian_cost_2
            
            guassian_cost_3 = self._calculate_gaussian(distribution, activations[index],3)
            total_guassian_3 +=guassian_cost_3
            
            median_cost = self._calculate_gaussian_median(distribution, activations[index])
            total_median +=median_cost
        return total_cost,total_guassian_1,total_guassian_2,total_guassian_3,total_median

    def _calculate_gaussian(self,distribution, simulated_data,dof):
        mean = distribution["Mean"]
        std = distribution["StandardDeviation"]
        peak_val = mean
    
        data = simulated_data[0]
        left = mean - (dof * std)
        right = mean + (dof * std)
    
        if data < left:
            g_cost = 1
        elif  data > right:
            g_cost = 1
        else:
            if data < mean:
                g_cost = (peak_val-data)/(peak_val-left)
                if np.isnan(g_cost):
                    # print('Less: data {} min {} max {} peak {} cost {}'.format(data,min_val,max_val,peak_val,anomalypercentage))
                    g_cost = 0
            else:
                g_cost = (data-peak_val)/(right-peak_val)
                if np.isnan(g_cost):
                    # print('Greater: data {} min {} max {} peak {} cost {}'.format(data,min_val,max_val,peak_val,anomalypercentage))
                    g_cost = 0
    
    
        return g_cost
    def _calculate_gaussian_median(self,distribution, simulated_data):
        min_val = distribution["Min"]
        max_val = distribution["Max"]
        median_val = distribution["Median"]

        # anomalypercentage = 0
        data = simulated_data[0]
        # print(data,min_val,max_val,peak_val)
        #for data in simulated_data:
        if data < min_val:
            anomalypercentage = 1
        elif  data > max_val:
            anomalypercentage = 1
        else:
            if data < median_val:
                anomalypercentage = (median_val-data)/(median_val-min_val)
                if np.isnan(anomalypercentage):
                    # print('Less: data {} min {} max {} peak {} cost {}'.format(data,min_val,max_val,peak_val,anomalypercentage))
                    anomalypercentage = 0
            else:
                anomalypercentage = (data-median_val)/(max_val-median_val)
                if np.isnan(anomalypercentage):
                    # print('Greater: data {} min {} max {} peak {} cost {}'.format(data,min_val,max_val,peak_val,anomalypercentage))
                    anomalypercentage = 0
        return anomalypercentage
    
    def _computeAnomaly(self,distribution, simulated_data):
        min_val = distribution["Min"]
        max_val = distribution["Max"]
        peak_val = distribution["Peak"]
    
        # anomalypercentage = 0
        data = simulated_data[0]
        #for data in simulated_data:
        if data < min_val:
            anomalypercentage = 1
        elif  data > max_val:
            anomalypercentage = 1
        else:
            if data < peak_val:
                anomalypercentage = (peak_val-data)/(peak_val-min_val)
                if np.isnan(anomalypercentage):
                    anomalypercentage = 0
            else:
                anomalypercentage = (data-peak_val)/(max_val-peak_val)
                if np.isnan(anomalypercentage):
                    anomalypercentage = 0
        return anomalypercentage
    def process_uncertainty(self,test_dataset_activation,test_label,train_activation_statistics):

        analysis_ds = []
        uncertainty_ds = []
        y_pred = test_label
        # for idx, data in enumerate(test_dataset):

        data = test_dataset_activation


        pred_class = y_pred
        total_cost,total_guassian_1,total_guassian_2,total_guassian_3,total_median = self._get_train_distribution_by_label(pred_class,train_activation_statistics,data)

        num_neurons = data.shape[0]
        tcsa = 1-(total_cost/num_neurons)

        total_median_rate = 1-(total_median/num_neurons)
        total_guassian_1_rate = 1-(total_guassian_1/num_neurons)
        total_guassian_2_rate = 1-(total_guassian_2/num_neurons)
        total_guassian_3_rate = 1-(total_guassian_3/num_neurons)

        analysis_ds.append([ y_pred,tcsa,total_guassian_1_rate,total_guassian_2_rate,total_guassian_3_rate,total_median_rate])
        analysis_df = pd.DataFrame(analysis_ds, columns=[
                                            'PredictedClass','Triangular','Gaussian1','Gaussian2','Gaussian3','Median'])
        uncertainty_ds.append([ total_cost,total_guassian_1,total_guassian_2,total_guassian_3,total_median])
        uncertainty_ds = pd.DataFrame(analysis_ds, columns=[
                                            'Triangular_Uncertainty','Gaussian1_Uncertainty','Gaussian2_Uncertainty','Gaussian3_Uncertainty','Median_Uncertainty'])    
        return analysis_df,uncertainty_ds    
    def process(self,test_dataset_activation,test_label,train_activation_statistics):

        analysis_ds = []
        y_pred = test_label
        # for idx, data in enumerate(test_dataset):

        data = test_dataset_activation


        pred_class = y_pred
        total_cost,total_guassian_1,total_guassian_2,total_guassian_3,total_median = self._get_train_distribution_by_label(pred_class,train_activation_statistics,data)

        num_neurons = data.shape[0]
        tcsa = 1-(total_cost/num_neurons)

        total_median_rate = 1-(total_median/num_neurons)
        total_guassian_1_rate = 1-(total_guassian_1/num_neurons)
        total_guassian_2_rate = 1-(total_guassian_2/num_neurons)
        total_guassian_3_rate = 1-(total_guassian_3/num_neurons)

        analysis_ds.append([ y_pred,tcsa,total_guassian_1_rate,total_guassian_2_rate,total_guassian_3_rate,total_median_rate])
        analysis_df = pd.DataFrame(analysis_ds, columns=[
                                            'PredictedClass','Triangular','Gaussian1','Gaussian2','Gaussian3','Median'])
        return analysis_df
    
    def _get_train_distribution_by_label(self, predicted_label, distribution_list, dataset): 

        for dist_list in distribution_list:
            if dist_list['ClassLabel'] == predicted_label:
                total_cost = 0
                props = dist_list['Distributions']
                total_cost,total_guassian_1,total_guassian_2,total_guassian_3,total_median = self._calculate_confidence_score(props,dataset)

        return total_cost,total_guassian_1,total_guassian_2,total_guassian_3,total_median
     
    def monte_carlo_dropout(self,test_ds, tf_model,max_loop=1):
        mcd_list = []
        for i in range(max_loop):
            mcd = tf_model(test_ds, training=True)
            mcd_max = tf.math.reduce_max(mcd).numpy()
            mcd_list.append(mcd_max)
        mcd_average = np.average(mcd_list)
        return mcd_average