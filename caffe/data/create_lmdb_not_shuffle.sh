#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=/home/code/wangfang/caffe-master/models/faceAttribute/data/imdb_wiki
DATA=/home/code/wangfang/caffe-master/data/imdb_wiki
TOOLS=build/tools

#TRAIN_DATA_ROOT=/home/code/wangfang/caffe-master/data/glasses/train_new/
#VAL_DATA_ROOT=/home/code/wangfang/caffe-master/data/lfwa/
TEST_DATA_ROOT=/home/code/wangfang/caffe-master/data/imdb_wiki/test/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
 RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
 RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

#if [ ! -d "$TRAIN_DATA_ROOT" ]; then
#  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
#  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet training data is stored."
#  exit 1
#fi

#if [ ! -d "$VAL_DATA_ROOT" ]; then
#  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
#  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet validation data is stored."
#  exit 1
#fi

#echo "Creating train lmdb..."

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    $TRAIN_DATA_ROOT \
#    $DATA/train_new.txt \
#    $EXAMPLE/glasses256_train_lmdb_not_shuffle \
#    1

#echo "Creating val lmdb..."

# GLOG_logtostderr=1 $TOOLS/convert_imageset \
#     --resize_height=$RESIZE_HEIGHT \
#     --resize_width=$RESIZE_WIDTH \
#     $VAL_DATA_ROOT \
#     $DATA/val_bald.txt \
#     $EXAMPLE/celeba_bald_val_lmdb_not_shuffle \
#     1

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $TEST_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/imdb_wiki_test_lmdb_not_shuffle \
    2

echo "Done."
