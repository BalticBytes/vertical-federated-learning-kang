# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 20:09:07 2021

@author: weikang
"""

import numpy as np
import tensorflow as tf


def cos_sim_attack(gradient, label):
    
    if len(label.shape) > 1:
        
        label = tf.reshape(label, [label.shape[0]*label.shape[1]])

    interval = 2000
    cos_sim_label = []
    for i in range(0, len(label), interval):
        cos_sim_array = np.dot(gradient[i:i+interval], np.transpose(gradient))
        cos_sim_array = sum(np.transpose(np.sign(cos_sim_array)))
        # print('cos_sim_array:', cos_sim_array)
        cos_sim_label += [0 if j > 0 else 1 for j in cos_sim_array]
    
    cos_sim_label = tf.cast(tf.reshape(cos_sim_label, [len(cos_sim_label), 1]), dtype=tf.float32)           
    
    mask = tf.equal(cos_sim_label, label)
    
    res = tf.where(mask)
    
    acc = sum([1 if label[i] == cos_sim_label[i] else 0 for i in range(len(label))])/len(label)

    return acc