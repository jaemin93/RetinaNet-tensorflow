'''
진행날짜: 2018-12-30 06:04 
진행시간: 4시간
진행상황: tfrecord로 읽는데 class 'bytes' 타입으로 읽어드림
진행계획: bytes에서 원하는 data type 으로 바꿔서 read data함수에 넘겨줘야함
'''


import os
import io
import tensorflow as tf
import numpy as np
from cv2 import imread, resize
import glob
import json
import matplotlib.pyplot as plt
from datasets.utils import anchor_targets_bbox, bbox_transform, padding, anchors_for_shape
import PIL.Image as pil
from object_detection.utils import dataset_util

TFRECORD_FILE = '2002_07_19_big_img_90.tfrecord'
IMAGE_FILE = 'img_18.jpg'

def _main():
    filename_queue = tf.train.string_input_producer([TFRECORD_FILE])
    encoded, x1, x2, y1, y2, label = read_tfrecord(filename_queue)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        vencoded, vx1, vx2, vy1, vy2, vlabel = sess.run([encoded, x1, x2, y1, y2, label])
        encoded_png_io = io.BytesIO(vencoded)
        image = pil.open(encoded_png_io)
        image = np.asarray(image)
        coord.request_stop()
        coord.join(threads)
        
    # im = imread(im_path)
    # im = np.array(im, dtype=np.float32)

    # image = Image.open(io.BytesIO(vencoded))
    # image.show()
    #images.append(im)
    #print(images)

        

def read_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    keys_to_features = {
        'img': tf.FixedLenFeature((), tf.string, default_value=''),
        'x1': tf.VarLenFeature(tf.float32),
        'x2': tf.VarLenFeature(tf.float32),
        'y1': tf.VarLenFeature(tf.float32),
        'y2': tf.VarLenFeature(tf.float32),
        'label': tf.FixedLenFeature((), tf.string, default_value='')
    }
    
    features = tf.parse_single_example(serialized_example,features= keys_to_features)
    
    encoded = tf.cast(features['img'],tf.string)
    x1 = tf.cast(features['x1'],tf.float32)
    x2 = tf.cast(features['x2'],tf.float32)
    y1 = tf.cast(features['y1'],tf.float32)
    y2 = tf.cast(features['y2'],tf.float32)
    label = tf.cast(features['label'],tf.string)
    
    return encoded, x1, x2, y1, y2, label


if __name__ == '__main__':
    _main()
