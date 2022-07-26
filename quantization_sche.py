# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:59:27 2021

@author: adamwei
"""
import tensorflow as tf
import numpy as np
import copy


def quant_recover_boundary(w, bucket_boundary):

    quant_update = copy.deepcopy(w)
    for i in range(len(bucket_boundary)-1):
        locations_bucket = (quant_update > bucket_boundary[i]) & (quant_update <= bucket_boundary[i+1])
        quant_update[locations_bucket] = (bucket_boundary[i] + bucket_boundary[i+1])/2
    
    return quant_update

class quant_process(object):
    def __init__(self, sketch_sche, values_update, quant_level, base_bits):
        
        self.sketch_sche = sketch_sche
        self.quant_level = quant_level
        self.base_bits = base_bits
        
        self.values_update_org = values_update
        
        self.values_shape = values_update.shape
        if len(values_update.shape) > 1:    
            self.values_update = tf.reshape(values_update, shape=[1,-1]).cpu().numpy().flatten()
        else:
            self.values_update = values_update
    
    def quant(self):
            
        
        if self.sketch_sche == 'bucket_uniform':
            
            min_value, max_value = min(self.values_update)-(1e-5), max(self.values_update)+(1e-5)
            
            _, uniform_bucket = np.histogram(self.values_update,
                                                bins=self.quant_level,
                                                range=[min_value, max_value],
                                                weights=None,
                                                density=False)
            quant_w = quant_recover_boundary(self.values_update, uniform_bucket)
            
            quant_w = tf.reshape(quant_w, shape = self.values_shape)
            
            
            
            communication_cost = self.base_bits * self.quant_level + np.ceil(np.log2(len(self.values_update)))

        
            
        else:
            
            print('\nNotice: no quantization')
            quant_w = self.values_update_org
            communication_cost = self.base_bits * len(self.values_update)
            
        mse_error = np.linalg.norm(quant_w-self.values_update_org, ord=2)
        
        # quant_incre = quant_w - self.values_update_org
        
        return quant_w, communication_cost, mse_error, uniform_bucket