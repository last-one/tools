import os
import sys
from caffe.proto import caffe_pb2

def toPrototxt(modelName, deployName):

    with open(modelName, 'rb') as f:
        caffemodel = caffe_pb2.NetParameter()
        caffemodel.ParseFromString(f.read())

    for item in caffemodel.layer:
        item.ClearField('blobs')

    with open(deployName, 'w') as f:
        f.write(str(caffemodel))

if __name__ == '__main__':

    toPrototxt(sys.argv[1], sys.argv[2])
