
# coding: utf-8

from Layers import *
import numpy as np


def conv_bn(bottom,
            kernel_h=1, kernel_w=1, kernel_output=96, 
            stride_h=1, stride_w=1,
            padding='SAME',
            use_bias=False,
            group=1,
            is_training=False,
            activation=tf.nn.relu,
            name=None):
    
    with tf.variable_scope(name) as scope:
        conv = conv_layer(bottom, kernel_h, kernel_w, kernel_output, stride_h, stride_w, padding, use_bias, group, 'conv')
        conv = batch_norm(conv, is_training, 'bn')
        if activation is not None:
            conv = activation(conv)

        return conv

def conv_pool(bottom,
              conv_kernel_h=1, conv_kernel_w=1, conv_kernel_output=96, 
              conv_stride_h=1, conv_stride_w=1,
              conv_padding='SAME',
              use_bias=False,
              group=1,
              activation=tf.nn.relu,
              pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=1, pool_stride_w=1, 
              pool_padding='SAME', pool_way='MAX',
              name=None):
    
    with tf.variable_scope(name) as scope:
        conv = conv_layer(bottom, conv_kernel_h, conv_kernel_w, conv_kernel_output, conv_stride_h, conv_padding, use_bias, group, 'conv')
        if activation is not None:
            conv = activation(conv)
        pool = pool_layer(conv, pool_kernel_h, pool_kernel_w, pool_stride_h, pool_stride_w, pool_padding, pool_way, 'pool')
        
        return pool

def conv_bn_pool(bottom,
                 conv_kernel_h=1, conv_kernel_w=1, conv_kernel_output=96, 
                 conv_stride_h=1, conv_stride_w=1,
                 conv_padding='SAME',
                 use_bias=False,
                 group=1,
                 is_training=False,
                 activation=tf.nn.relu,
                 pool_kernel_h=2, pool_kernel_w=2, pool_stride_h=1, pool_stride_w=1, 
                 pool_padding='SAME', pool_way='MAX',
                 name=None):
    
    with tf.variable_scope(name) as scope:
        conv = conv_bn(bottom, conv_kernel_h, conv_kernel_w, conv_kernel_output, 
                conv_stride_h, conv_stride_w, conv_padding, use_bias, group, is_training,
                activation, scope.name)
        pool = pool_layer(conv, pool_kernel_h, pool_kernel_w, pool_stride_h, pool_stride_w, pool_padding, pool_way, 'pool')
        
        return pool

def fc_dropout(bottom, output_num=4096, activation=tf.nn.relu, dropout_keep_prob=0.5, name=None):
    
    with tf.variable_scope(name) as scope:
        fc = fc_layer(bottom, output_num, activation, 'fc')
        fc = dropout(fc, dropout_keep_prob, name='dropout')
        
        return fc

