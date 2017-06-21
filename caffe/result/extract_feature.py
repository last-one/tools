import sys
import os
import numpy as np
import argparse
import caffe

#np.set_printoptions(threshold='nan')

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prototxt', dest='prototxt', help='the path of prototxt')
    parser.add_argument('-n', '--net', dest='model', help='the path of caffemodel')
    parser.add_argument('-b', '--blob', dest='blob', help='the name of layer extracred from')
    parser.add_argument('-i', '--input', dest='input_path', help='the path of input')
    parser.add_argument('-o', '--output', dest='output_path', help='the path of output')
    parser.add_argument('-g', '--gpu', dest='gpu_id', help='gpu_id', type=int, default=0)
    parser.add_argument('-m', '--mean', dest='mean_file', type=str, default=None)
    parser.add_argument('-c', '--crop', dest='crop_size', type=int, default=227)
    args = parser.parse_args()

    return args

def get_transformer(net, mean_file):

    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(mean_file, 'rb').read())
    mean_np = caffe.io.blobproto_to_array(mean_blob)
    mean_np = mean_np[0]

    transformer = caffe.io.Transformer({'data': (1, 3, 256, 256)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean_np)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    return transformer

def crop(img, crop_size):

    h = img.shape[1]
    w = img.shape[2]
    if h <= crop_size:
        sh = 0
        eh = h
    else:
        sh = (h - crop_size) / 2
        eh = sh + crop_size
    if w <= crop_size:
        sw = 0
        ew = w
    else:
        sw = (w - crop_size) / 2
        ew = sw + crop_size
    crop_img = img[:, sh: eh, sw: ew]

    return crop_img

def get_feature(net, transformer, args):

    net.blobs['data'].reshape(1, 3, 227, 227)

    for picture in os.listdir(args.input_path):
        name = picture.split('.')[0]
        picture_path = os.path.join(args.input_path, picture)
        img = caffe.io.load_image(picture_path)
        tr_img = transformer.preprocess('data', img)
        crop_img = crop(tr_img, args.crop_size)

        net.blobs['data'].data[...] = crop_img
        net.forward()
        feat = net.blobs[args.blob].data[0]

        feat_path = os.path.join(args.output_path, name + '.npy')
        np.save(feat_path, feat)

if __name__ == '__main__':

    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    transformer = get_transformer(net, args.mean_file)
    get_feature(net, transformer, args)
