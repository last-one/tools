#!/usr/bin/env sh

# [the path of model]    [the path of test_prototxt]  [layer name] [output path] [all size / batch_size] [gpu id]

/home/hypan/workspace/caffe/build/tools/extract_features_binary /home/hypan/workspace/caffe/models/faceAttribute/celebA/resnet101_v2/models/celeba_resnet101_v2_finetune__iter_170000.caffemodel /home/hypan/workspace/caffe/models/faceAttribute/celebA/resnet101_v2/test.prototxt classifier_40 /home/hypan/workspace/caffe/models/faceAttribute/celebA/resnet101_v2/features/celeba_test_170000.dat 640 0 
