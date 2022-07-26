# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:33:51 2021

@author: adamwei
"""

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense


class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.activation = activation

        self.hidden_layer = [Dense(i, activation=self.activation, kernel_regularizer=tf.keras.regularizers.L2(0.01))
                             for i in self.hidden_units]
        self.output_layer = Dense(self.output_dim, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output
    

class tf_organization_graph(Model):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()

        self.Dense = Dense_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        client_output = self.Dense(inputs)
        return client_output


class tf_top_graph(Model):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.Dense = Dense_layer(hidden_units, output_dim, activation)

    # Ming modified the call function on 12/11/2021 to make it more generic 
    def call(self, client_intputs):

        x = tf.concat([client_intputs[0], client_intputs[1]], axis=-1)
        
        if len(client_intputs) > 2:
            for input_idx in range(len(client_intputs)-2):
                x = tf.concat([x, client_intputs[input_idx+2]], axis=-1)
            
        output = self.Dense(x)
        output = tf.nn.softmax(output)
        output = tf.reshape(output, shape = [len(output),-1])
        return output
    
    # def call(self, client_intput0, client_intput1, client_intput2):

    #     x = tf.concat([client_intput0, client_intput1], axis=-1)
    #     x = tf.concat([x, client_intput2], axis=-1)
    #     output = self.Dense(x)
    #     output = tf.nn.sigmoid(output)
    #     output = tf.reshape(output, shape = [len(output),-1])
    #     return output
    
class tf_graph(Model):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.Dense = Sequential()
        for i in hidden_units:
            self.Dense.add(Dense(i,
                                 activation=activation,
                                 kernel_regularizer=tf.keras.regularizers.L2(0.1)))
        self.Dense.add(Dense(output_dim,
                             activation=None,
                             kernel_regularizer=tf.keras.regularizers.L2(0.1)))
        # self.Dense = Dense_layer(hidden_units, output_dim, activation)

    def call(self, inputs):

        output = self.Dense(inputs)
        # output = tf.nn.softmax(output)

        return output
