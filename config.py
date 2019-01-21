import os
# directory config
DATA_DIR = 'C:\\Users\\iceba\\develop\\data'
ROOT_DIR = 'C:\\Users\\iceba\\develop\\data\\face'
SAVE_DIR = 'C:\\Users\\iceba\\develop\\data\\retinanet_ckpt'
IMAGE = 'originalPics'
TRAIN = 'train'
TFRECORD = 'face_tfrecords'
LABEL = 'FDDB-folds'
PRETRAINED_DIR = 'pretrained_resnet'
PRETRAINED_MODEL = 'resnet_v2_50'
DATA_BATCH_SIZE = 100

# tfrecords_list
TF_LIST = list()

for info in os.listdir(os.path.join(ROOT_DIR, TFRECORD)):
    TF_LIST.append(os.path.join(ROOT_DIR + os.sep + TFRECORD, info))

