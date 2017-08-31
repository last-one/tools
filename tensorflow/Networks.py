
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import sys
from Blocks import *


class Networks(object):
    
    def __init__(self, x, is_training,
                 dropout_keep_prob, skip_layers=[], num_classes=1000):
        
        self.X = x
        self.dropout_keep_prob = dropout_keep_prob
        self.num_classes = num_classes
        self.is_training = is_training
        self.skip_layers = skip_layers
    
    def build(self):
        
        pass
    
    def load_model(self, sess, model_path):
        
        if model_path[-5:] == '.ckpt':
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
        elif model_path[-5:] == '.npy':
            weights_dict = np.load(model_path, encoding = 'bytes').item()
        
            # Loop over all layer names stored in the weights dict
            for op_name in weights_dict:
                # Check if the layer is one of the layers that should be reinitialized
                if op_name not in self.skip_layers:
                    with tf.variable_scope(op_name, reuse = True):
                        # Loop over list of weights/biases and assign them to their corresponding tf variable
                        for data in weights_dict[op_name]:
                            # Biases
                            if len(data.shape) == 1:
                                var = tf.get_variable('biases', trainable = True)
                                session.run(var.assign(data))
                            # Weights
                            else:
                                var = tf.get_variable('weights', trainable = True)
                                session.run(var.assign(data))
        else:
            pass
        
class AlexNet(Networks):
    
    def build(self):
        
        end_points = {}
        
        conv1 = conv_bn_pool(self.X, 11, 11, 96, 4, 4, 'VALID', False, 1, self.is_training, tf.nn.relu,
                           3, 3, 2, 2, 'VALID', 'MAX', 'conv1')
        end_points['conv1'] = conv1
        
        conv2 = conv_bn_pool(conv1, 5, 5, 256, 1, 1, 'SAME', False, 2, self.is_training, tf.nn.relu,
                           3, 3, 2, 2, 'VALID', 'MAX', 'conv2')
        end_points['conv2'] = conv2
        
        conv3 = conv_bn(conv2, 3, 3, 384, 1, 1, 'SAME', False, 2, self.is_training, tf.nn.relu, 'conv3')     
        end_points['conv3'] = conv3
        
        conv4 = conv_bn(conv3, 3, 3, 384, 1, 1, 'SAME', False, 2, self.is_training, tf.nn.relu, 'conv4')     
        end_points['conv4'] = conv4
        
        conv5 = conv_bn_pool(conv4, 3, 3, 256, 1, 1, 'SAME', False, 1, self.is_training, tf.nn.relu,
                           3, 3, 2, 2, 'VALID', 'MAX', 'conv5')
        end_points['conv5'] = conv5
  
        fc6 = fc_dropout(conv5, 4096, tf.nn.relu, self.dropout_keep_prob, 'fc6')
        end_points['fc6'] = fc6

        fc7 = fc_dropout(fc6, 4096, tf.nn.relu, self.dropout_keep_prob, 'fc7')
        end_points['fc7'] = fc7
        
        fc8 = fc_layer(fc7, self.num_classes, tf.nn.relu, 'fc8')
        end_points['fc8'] = fc8
        
        return end_points

class VGG16(Networks):
    
    def build(self):
        
        pass
    
