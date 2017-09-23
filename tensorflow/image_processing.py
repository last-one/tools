# coding: utf-8

import tensorflow as tf
import numpy as np
import os
from Networks import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from Layers import *


def load_data(annotation_path):
    
    filenames = []
    labels = []
    with open(annotation_path, 'r') as fp:
        for line in fp:
            info = line.strip().split(' ')
            filenames.append(info[0])
            labels.append(int(info[1]))
    print filenames, labels
    return filenames, labels

def load_input(annotation_path, crop_w, crop_h, batch_size=32, capacity=64, is_training=True):
    
    filenames, labels = load_data(annotation_path)
    filenames, labels = tf.train.slice_input_producer([filenames, labels], shuffle=is_training)
    images_contents = tf.read_file(filenames)
    images = tf.image.decode_jpeg(images_contents, channels=3)
    
    images = tf.image.resize_image_with_crop_or_pad(images, crop_w, crop_h)
    
    image_batch, label_batch = tf.train.batch([images, labels],
                                                batch_size= batch_size,
                                                num_threads= 32, 
                                                capacity = capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch

def show(image, label, batch_size, iter_size):
    
    for j in range(batch_size):
        print image[j].shape
        img = np.array(image[j,:,:,:], dtype=np.uint8)
        img11 = Image.fromarray(img)  
        #img[...,0], img[...,2] = img[...,2], img[...,0]
        #print img
        #img = img.transpose((2,0,1))
        print img.shape
        img11.save("images/{}_{}.jpg".format(label[j], iter_size))
        print "name is {}".format(label[j])
    print '----------------------------------------------'

def train_validation_queue(train_list, validation_list):
    
    is_training = tf.placeholder('bool', [], name='is_training')
    drop_keep_prob = tf.placeholder(tf.float32, [], name='drop_keep_prob')
    
    train_image, train_label = load_input(train_list, 227, 227, 4)
    val_image, val_label = load_input(validation_list, 227, 227, 2)
    data = tf.cond(is_training, lambda: train_image, lambda: val_image)
    labels = tf.cond(is_training, lambda: train_label, lambda: val_label)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.02)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    testnet = TestNet(x, is_training, drop_keep_prob)
    end_points = testnet.build()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        i = 0
        max_iter = 20
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < max_iter:
                image_batch, label_batch = sess.run([data, labels], feed_dict={is_training: True})
                out1 = sess.run(end_points['data'], feed_dict={x: image_batch, is_training: True, drop_keep_prob:0.5})
                show(image_batch, label_batch, 4, i)
                show(out1, label_batch, 4, i*100)
                i+=1
                if i % 2 != 0:
                    image_batch, label_batch = sess.run([data, labels], feed_dict={is_training: False})
                    out1 = sess.run(end_points['data'], feed_dict={x: image_batch, is_training: False, drop_keep_prob:1})
                    show(image_batch, label_batch, 2, i)
                    show(out1, label_batch, 2, i*100)
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
            coord.join(threads)

train_validation_queue(train_list='train.txt', validation_list='test.txt')
