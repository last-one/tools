#!/usr/bin/env sh

# [the path of model]    [the path of test_prototxt]  [layer name] [output path] [all size / batch_size] [gpu id]

/home/hypan/workspace/caffe/build/tools/extract_features_binary /home/hypan/workspace/caffe/models/faceAttribute/celebA/caffenet/models/celeba_finetune_iter_160000.caffemodel /home/hypan/workspace/caffe/models/faceAttribute/celebA/caffenet/test.prototxt fc8_celeba /home/hypan/workspace/caffe/models/faceAttribute/celebA/caffenet/features/celeba_test_160000.dat 640 3 
