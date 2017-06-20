#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

DATA=$1
DIRPATH=$2
TOOLS=/home/hypan/caffe/build/tools

$TOOLS/compute_image_mean $DATA/train_lmdb_shuffle \
  $DIRPATH/train.binaryproto

echo "Done."
