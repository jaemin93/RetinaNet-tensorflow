import os
import numpy as np
from cv2 import imread, resize
import numpy as np
import glob
import PIL.Image as pil
import io
import json
import tensorflow as tf
from datasets.utils import anchor_targets_bbox, bbox_transform, padding, anchors_for_shape
from config import *

IM_EXTENSIONS = ['png', 'jpg', 'bmp']


def read_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    keys_to_features = {
        'img': tf.FixedLenFeature((), tf.string, default_value=''),
        'idx': tf.VarLenFeature(tf.int64),
        'height': tf.VarLenFeature(tf.int64),
        'width': tf.VarLenFeature(tf.int64),
        'x1': tf.VarLenFeature(tf.float32),
        'x2': tf.VarLenFeature(tf.float32),
        'y1': tf.VarLenFeature(tf.float32),
        'y2': tf.VarLenFeature(tf.float32),
        'label': tf.FixedLenFeature((), tf.string, default_value='')
    }
    
    features = tf.parse_single_example(serialized_example,features= keys_to_features)
    
    encoded = tf.cast(features['img'],tf.string)
    idx = tf.cast(features['idx'],tf.int64)
    h = tf.cast(features['height'],tf.int64)
    w = tf.cast(features['width'],tf.int64)
    x1 = tf.cast(features['x1'],tf.float32)
    x2 = tf.cast(features['x2'],tf.float32)
    y1 = tf.cast(features['y1'],tf.float32)
    y2 = tf.cast(features['y2'],tf.float32)
    label = tf.cast(features['label'],tf.string)
    # print(type(label), label.value)
    return encoded, idx, h, w, x1, x2, y1, y2, label

def read_data(train_set, image_size, start, end, no_label=False):
    """
    Load the data and preprocessing for RetinaNet detector
    :param data_dir: str, path to the directory to read.
                     It should include class_map, annotations
    :image_size: tuple, image size for resizing images
    :no_label: bool, whetehr to load labels
    :return: X_set: np.ndarray, shape: (N, H, W, C).
             y_set: np.ndarray, shape: (N, N_box, 5+num_classes).
    """
    #jaemin`s code
    
    anchors = anchors_for_shape(image_size)
    ih, iw = image_size
    images = []
    labels = []

    # jaemin`s code
    filename_queue = tf.train.string_input_producer(train_set[start:end])
    encoded, index, h, w, x1, x2, y1, y2, label = read_tfrecord(filename_queue)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        index_list = list()
        try:
            count = 1
            while not coord.should_stop():
                vencoded, vindex, vh, vw, vx1, vx2, vy1, vy2, vlabel = sess.run([encoded, index, h, w, x1, x2, y1, y2, label])
                if count % 100 is 0:
                    print('총 2845:',count)
                count += 1
                if vindex in index_list:
                    coord.should_stop()
                    break
                else:
                    index_list.append(vindex)
                
                encoded_png_io = io.BytesIO(vencoded)
                image = pil.open(encoded_png_io)
                im = np.array(image, dtype=np.float32)
                # load image and resize image
                im_original_sizes = im.shape[:2]
                im = resize(im, (image_size[1], image_size[0]))
                if len(im.shape) == 2:
                    im = np.expand_dims(im, 2)
                    im = np.concatenate([im, im, im], -1)
                images.append(im)
                # print(images)
                if no_label:
                    labels.append(0)
                    continue
                # load bboxes and reshape for retina model
                bboxes = list()
                for c_idx, c_name in enumerate(range(1)):   # num_calsses set
                    for i in range(len(vx1.values)):
                        oh, ow = im_original_sizes
                        x_min, y_min, x_max, y_max = vx1.values[i] / ow, vy1.values[i] / oh, vx2.values[i] / ow, vy2.values[i] / oh
                        # print([x_min, y_min, x_max, y_max, int(c_idx)+1])
                        bboxes.append([x_min, y_min, x_max, y_max, int(c_idx)+1])
                
                    bboxes = np.array(bboxes)
                    # print(npbboxes.shape)
                    bboxes = np.array([iw, ih, iw, ih, 1], dtype=np.float32) * bboxes
                    # print(npbboxes.shape)
                    b_labels, annotations = anchor_targets_bbox(im.shape, bboxes, 1+1, anchors) # num_calsses set
                    # print(b_labels.shape, annotations.shape)
                    # print(annotations)
                    regression = bbox_transform(anchors, annotations)
                    n_label = np.array(np.append(regression, b_labels, axis=1), dtype=np.float32)
                    # print(len(regression), len(b_labels), sep='\n')
                labels.append(n_label)

        except:
            print('error')
        finally:
            X_set = np.array(images, dtype=np.float32)
            y_set = np.array(labels, dtype=np.float32)
            coord.request_stop()
            coord.join(threads)

    return X_set, y_set

def load_json(json_path):
    """
    Load json file
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


class DataSet(object):

    def __init__(self, images, labels=None):
        """
        Construct a new DataSet object.
        :param images: np.ndarray, shape: (N, H, W, C)
        :param labels: np.ndarray, shape:
        """
        if labels is not None:
            assert images.shape[0] == labels.shape[0],\
                ('Number of examples mismatch, between images and labels')
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels  # NOTE: this can be None, if not given.
        # image/label indices(can be permuted)
        self._indices = np.arange(self._num_examples, dtype=np.uint)
        self._reset()

    def _reset(self):
        """Reset some variables."""
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def sample_batch(self, batch_size, shuffle=True):
        """
        Return sample examples from this dataset.
        :param batch_size: int, size of a sample batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes)
        """
        if shuffle:
            indices = np.random.choice(self._num_examples, batch_size)
        else:
            indices = np.arange(batch_size)
        batch_images = self._images[indices]
        if self._labels is not None:
            batch_labels = self._labels[indices]
        else:
            batch_labels = None
        return batch_images, batch_labels

    def next_batch(self, batch_size, shuffle=True):
        """
        Return the next 'batch_size' examples from this dataset.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes)
        """

        start_index = self._index_in_epoch

        # Shuffle the dataset, for the first epoch
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # Go to the next epoch, if current index goes beyond the total number
        # of examples
        if start_index + batch_size > self._num_examples:
            # Increment the number of epochs completed
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # Shuffle the dataset, after finishing a single epoch
            if shuffle:
                np.random.shuffle(self._indices)

            # Start the next epoch
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self._images[indices_rest_part]
            images_new_part = self._images[indices_new_part]
            batch_images = np.concatenate(
                (images_rest_part, images_new_part), axis=0)
            if self._labels is not None:
                labels_rest_part = self._labels[indices_rest_part]
                labels_new_part = self._labels[indices_new_part]
                batch_labels = np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self._images[indices]
            if self._labels is not None:
                batch_labels = self._labels[indices]
            else:
                batch_labels = None

        return batch_images, batch_labels
