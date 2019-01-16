# RetinaNet-tensorflow
Object-detection-tutorials with Retina (tensorflow)

## Requirements
Python 3.6 (python 3.5 is not working object detection api)
```
numpy==1.15.1
matplotlib==2.0.2
opencv-python==3.3.0.10
tensorflow-gpu == 1.10.1
object detection api for make tfrecords: [link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) 

```

toy dataset: [FDDB: Face Detection Data Set and Benchmark(FDDB)](http://vis-www.cs.umass.edu/fddb/)

ellipsis_to_rectangle.py script helps to make annotations file for detection

## DEVELOP GOAL
Convert image(.jpg), label(.anno) data to tfrecords format and read tfrecords

STEP
```
1: Convert data (tfrecord): Done
2: Read tfrecord: Done
3: Synchronization: Done
```
## How can I run tfrecord_train.py?


v.0.1: STEP
```
1. convert face_data_convert_tfrecords.py to yours and run face_data_convert_tfrecords.py in utils:
    - line 14, 15: read pics and anno

2. convert tfrecord_train.py to yours:
    - line 10, 11: read your {}.tfrecords directory
    - line 51: convert save_dir to yours

3. convert optimizers.py to yours in learning:  
    - line 89: convert num_eval to yours  

4. convert retina.py to yours in models:
    - line 51: you make dir pretrained_models and that have pretrained ckpt file, write path
            file: [resnet_v2_50](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)
```

***
v.0.2: __configuration you directory in config.py__
directory structure:

            data
            ├─face
            │  ├─face_tfrecords
            │  ├─face_tfrecords2
            │  ├─FDDB-folds
            │  ├─FDDB-list
            │  └─originalPics
            ├─pretrained_resnet
            └─retinanet_ckpt
                └─log
                    ├─train
                    └─val


pleases write e-mail(icebanana93@gmail.com) not if you success.