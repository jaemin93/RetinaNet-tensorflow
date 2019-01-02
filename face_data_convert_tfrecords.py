'''
Code by jaemin93 (https://github.com/jaemin93)

'''

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import dataset_util

IMG_ROOT_DIR = 'RetinaNet-tensorflow\\data\\face\\originalPics'
LABEL_ROOT_DIR = 'RetinaNet-tensorflow\\data\\face\\FDDB-folds'
TFRECORD_ROOT_DIR = 'RetinaNet-tensorflow\\data\\face\\tmp2'

def _main():
    print('Make datasets list....')
    image_list = make_file_list(IMG_ROOT_DIR)
    print('Done')
    print('Convert datasets list to Dict type')
    d = make_label_list(LABEL_ROOT_DIR, image_list)
    print('Done')
    print('Converting Dict{img:label} to tfrecords format...')
    for file_name, values in d.items():
        img_name = file_name.split('/')
        img_name = '_'.join(img_name)
        output_path = TFRECORD_ROOT_DIR + os.sep + img_name[:-4] + str('.tfrecord')
        img_tfrecord = create_tfrecord(file_name, values)
        writer = tf.python_io.TFRecordWriter(output_path)
        writer.write(img_tfrecord.SerializeToString())
    print('Done')


def create_tfrecord(file_name, features):
    '''
        file_name = FDDB-fold-0x-ellipseList.txt에 있는 이미지 이름 + '.jpg' 확장자 까지 붙은것
        features = 해당 이미지 이름을 키로 가지고있는 ellipse 정보들 아래는 예
        [['69.371552', '44.246872', '-1.492985', '223.325063', '126.940414', '1']]
    '''
    img_path = IMG_ROOT_DIR + os.sep + os.sep.join(file_name.split('/'))

    img = read_imagebytes(img_path)
    label_text = b'face'
    x1_list = list()
    x2_list = list() 
    y1_list = list() 
    y2_list = list()

    for num_cls_idx in range(len(features)):
        rad = abs(float(features[num_cls_idx][2]))
        h = float(features[num_cls_idx][0]) * np.sin(rad)
        w = float(features[num_cls_idx][1]) * np.sin(rad)
        x1, x2 = float(features[num_cls_idx][3]) - w, float(features[num_cls_idx][3]) + w
        y1, y2 = float(features[num_cls_idx][4]) - h, float(features[num_cls_idx][4]) + h

        x1_list.append(x1)
        x2_list.append(x2)
        y1_list.append(y1)
        y2_list.append(y2)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'img': dataset_util.bytes_feature(img),
        'x1': dataset_util.float_list_feature(x1_list),
        'x2': dataset_util.float_list_feature(x2_list),
        'y1': dataset_util.float_list_feature(y1_list),
        'y2': dataset_util.float_list_feature(y2_list),
        'label': dataset_util.bytes_feature(label_text),
    }))

    return tf_example

def read_imagebytes(imagefile):
    file = open(imagefile,'rb')
    bytes = file.read()
    return bytes

def make_label_list(path, img_list):
    info = dict()
    for label_file_name in os.listdir(path):
        info_key = label_file_name+'.jpg'
        label_dir = path + os.sep + label_file_name
        with open(label_dir, 'r', encoding='utf-8') as f_l:
            while True:
                coordinate = list()
                line = f_l.readline()
                original_img_name = line.split('\n')[0] + '.jpg'
                img_name = line[line.find('_')-3:]
                if not img_name:
                    break
                num_anchors = int(f_l.readline())
                for i in range(num_anchors):
                    coordinate.append(f_l.readline().split())
                info[original_img_name] = coordinate

    return info


def make_file_list(path):
    file_list = list()
    for year in os.listdir(path):
        YEAR_ROOT_DIR = IMG_ROOT_DIR + os.sep + str(year)
        for month in os.listdir(YEAR_ROOT_DIR):
            MONTH_ROOT_DIR = YEAR_ROOT_DIR + os.sep + str(month)
            for day in os.listdir(MONTH_ROOT_DIR):
                DAY_ROOT_DIR = MONTH_ROOT_DIR + os.sep + str(day)
                for classifier in os.listdir(DAY_ROOT_DIR):
                    CLASSIFIER_ROOT_DIR = DAY_ROOT_DIR + \
                        os.sep + str(classifier)
                    for file_idx, file_name in enumerate(os.listdir(CLASSIFIER_ROOT_DIR)):
                        src = str(year) + '_' + str(month) + '_' + str(day) + '_' + str(classifier) \
                            + '_' + str(file_name)
                        file_list.append(src[:-4])

    return file_list


if __name__ == '__main__':
    _main()
