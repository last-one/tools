# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.1
UPDATE_OPS_COLLECTION = 'update_ops'  # must be grouped with training op

def _get_variable(name,
                 shape,
                 initializer,
                 weight_decay=0.0,
                 dtype='float',
                 trainable=True):
    
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    
    return tf.get_variable(name,
                          shape=shape,
                          initializer=initializer,
                          dtype=dtype,
                          regularizer=regularizer,
                          trainable=trainable)

def conv_layer(bottom,
               kernel_h=1, kernel_w=1, kernel_output=96, 
               stride_h=1, stride_w=1,
               padding='SAME',
               use_bias=False,
               group=1,
               name=None):
    
    with tf.variable_scope(name) as scope:
        
        channels = int(bottom.get_shape()[-1])
        
        initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
        weight = _get_variable(name = 'weights',
                               shape = [kernel_h, kernel_w, channels/group, kernel_output],
                               initializer=initializer,
                               weight_decay=CONV_WEIGHT_DECAY)
        if group > 1:
            x = tf.split(value = bottom, num_or_size_splits = group, axis = 3)
            w = tf.split(value = weight, num_or_size_splits = group, axis = 3)
            feature = [tf.nn.conv2d(t1, t2, [1, stride_h, stride_w, 1], padding=padding) for t1, t2 in zip(x, w)]  
            conv = tf.concat(axis = 3, values = feature)
        else:
            conv = tf.nn.conv2d(bottom, weight, [1, stride_h, stride_w, 1], padding=padding)
        
        
        if use_bias:
            bias = _get_variable(name = 'biases',
                                 shape = [kernel_output],
                                 initializer=tf.zeros_initializer)
            conv = tf.nn.bias_add(conv, bias)
            
        return conv

def pool_layer(bottom, kernel_h=2, kernel_w=2, stride_h=1, stride_w=1, 
                    padding='SAME', way='MAX', name=None):
    
    if way == 'MAX':
        return tf.nn.max_pool(bottom, ksize = [1, kernel_h, kernel_w, 1],
                       strides = [1, stride_h, stride_w, 1], 
                       padding=padding, name = name)
    elif way == 'AVE':
        return tf.nn.avg_pool(bottom, ksize = [1, kernel_h, kernel_w, 1],
                       strides = [1, stride_h, stride_w, 1],
                       padding=padding, name = name)
    else:
        return None

def fc_layer(bottom, output_num=4096, activation=tf.nn.relu, name=None):
    
    input_num = bottom.get_shape()
    if len(input_num) != 2:
        dim = 1
        for d in input_num[1:]:
            dim *= int(d)
        bottom = tf.reshape(bottom, [-1, dim])
    else:
        dim = input_num[1]
    
    with tf.variable_scope(name) as scope:
        
        weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
        
        weight = _get_variable('weights',
                               shape=[dim, output_num],
                               initializer=weights_initializer,
                               weight_decay=FC_WEIGHT_DECAY)
        
        bias = _get_variable('biases',
                             shape=[output_num],
                             initializer=tf.zeros_initializer)
        
        fc = tf.nn.xw_plus_b(bottom, weight, bias, name=name)
        
        if activation is not None:
            fc = activation(fc)

        return fc

def dropout(bottom, dropout_keep_prob=0.5, name=None):
    
    return tf.nn.dropout(bottom, dropout_keep_prob, name=name)

def batch_norm(bottom, is_training=False, name=None):
    
    with tf.variable_scope(name) as scope:
        x_shape = bottom.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))

        beta = _get_variable('beta',
                            params_shape,
                            initializer=tf.zeros_initializer)
        gamma = _get_variable('gamma',
                             params_shape,
                             initializer=tf.ones_initializer)

        moving_mean = _get_variable('moving_mean',
                                   params_shape,
                                   initializer=tf.zeros_initializer,
                                   trainable=False)
        moving_variance = _get_variable('moving_variance',
                                       params_shape,
                                       initializer=tf.ones_initializer,
                                       trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(bottom, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                  mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance,
                                                                      variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            tf.convert_to_tensor(is_training, dtype='bool', name='is_training'),
            lambda: (mean, variance),
            lambda:(moving_mean, moving_variance))

        bn = tf.nn.batch_normalization(bottom, mean, variance, beta, gamma, BN_EPSILON)
    
        return bn

def concat(bottoms, axis=1, name=None):
    
    return tf.concat(bottoms, axis=axis, name=name)

def loss(logits, labels):
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
 
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_loss)
    tf.summary.scalar('loss', loss_)

    return loss_

