import os
import numpy as np
import tensorflow as tf
from datasets import data as dataset
from models.retina import RetinaNet as ConvNet
from learning.optimizers import AdamOptimizer as Optimizer
from learning.evaluators import RecallEvaluator as Evaluator
from config import *

""" 1. Load and split datasets """
root_dir = ROOT_DIR # FIXME
trainval_dir = os.path.join(root_dir, TFRECORD)

# Set image size and number of class
IM_SIZE = (512, 512)
NUM_CLASSES = 1
start = 0
end = start + DATA_BATCH_SIZE
""" 2. Set training hyperparameters"""
hp_d = dict()

# FIXME: Training hyperparameters
hp_d['batch_size'] = 8
hp_d['num_epochs'] = 50
hp_d['init_learning_rate'] = 1e-4
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 10
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8
hp_d['score_threshold'] = 1e-4
hp_d['nms_flag'] = True
hp_d['pretrain'] = True

""" 3. Build graph, initialize a session and start training """
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, None)

evaluator = Evaluator()

# Load trainval set and split into train/val sets

def data_iterator(start, end):
    X_trainval_batch, y_trainval_batch = dataset.read_data(IM_SIZE, start, end)
    return X_trainval_batch, y_trainval_batch, batch_idx

def data_shuffle(data):
    idxs = np.arange(0, len(data))
    np.random.shuffle(idxs)
    shuf_data = data[idxs]
    return shuf_data

sess = tf.Session(graph=graph, config=config)
shuf_data = data_shuffle(TF_LIST)

for step in range(500):
    if start + DATA_BATCH_SIZE > len(TF_LIST):
        X_trainval, y_trainval idx = data_iterator(shuf_data, start, len(TF_LIST)-1)
        start = 0
        shuf_data = data_shuffle(TF_LIST)
    else:
        X_trainval, y_trainval idx = data_iterator(shuf_data, start, start+DATA_BATCH_SIZE)
        start = start+DATA_BATCH_SIZE
    print('read data mini batch success')
    trainval_size = X_trainval.shape[0]
    val_size = int(trainval_size * 0.1) # FIXME
    val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
    train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

    optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

    train_results = optimizer.train(sess, save_dir=SAVE_DIR, details=True, verbose=True, **hp_d)
