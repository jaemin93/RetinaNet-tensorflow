import os
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2 as resnet_v2
from datasets.utils import anchors_for_shape
from models.layers import build_head_cls, build_head_loc, conv_layer, resize_to_target
from models.utils import smooth_l1_loss, focal_loss, bbox_transform_inv
from models.nn import DetectNet

slim = tf.contrib.slim
# RetinaNet 클래스 선언
# DetectNet이라는 Class를 받는다.(DetectNet은 ABCMeta를 받는 Class이다.)
# __init__, _build_model, _build_loss함수로 구성됨
# 역할 : 네트웍모델을 구축
# train.py에서는 ConvNet으로 import naming됨
class RetinaNet(DetectNet):
    """RetinaNet Class"""
    # 인스턴스를 만들때 함께 생성해줘야하는 파라메터값들을 정의함.
    # 새로운 인스턴스가 만들어 질때 __init__ 함수가 호출됨
    # 파라미터의 의미들은 생각해봐야함!!
    # kwargs : 딕셔너리,아래 코드에서 **는 딕셔너리 타입을 unpacking 할때 사용
    # key와 value를 argument값으로 넘겨준다.
    # anchors: Bounding box에 후보로 사용되는 상자
    def __init__(self, input_shape, num_classes, anchors=None, **kwargs):
        # anchors 는 input의 [0~1] 의 shape을 가진다.
        self.anchors = anchors_for_shape(input_shape[:2]) if anchors is None else anchors
        # label 형태: [None, self.anchors.shape[0], 5 + num_classes]
        self.y = tf.placeholder(tf.float32, [None, self.anchors.shape[0], 5 + num_classes])
        super(RetinaNet, self).__init__(input_shape, num_classes, **kwargs)
        # self.pred_y = self._build_pred_y(self)
        self.pred_y = self.pred
    
    # 모델 빌드하는 함수
    # kwargs : 딕셔너리,아래 코드에서 **는 딕셔너리 타입을 unpacking 할때 사용
    # key와 value를 argument값으로 넘겨준다.
    def _build_model(self, **kwargs):
        # d라는 딕셔너리 생성
        d = dict()
        # 분류 갯수 값 대입
        num_classes = self.num_classes
        # pop함수로 unpacking된 kwargs중에 'pretrain' key가 없으면 True값을 리턴
        pretrain = kwargs.pop('pretrain', True)
        # pop함수로 unpacking된 kwargs중에 'frontend' key가 없으면, resnet_v2_50 key의 value가 리턴
        frontend = kwargs.pop('frontend', 'resnet_v2_50')
        # pop함수로 unpacking된 kwargs중에 num_anchors key에서 value를 꺼낸다.
        # kwargs에 num_anchors키가 없으면 9로 리턴된다.
        # models/utils.py line60: num_anchors = len(ratios) * len(scales)
        num_anchors = kwargs.pop('num_anchors', 9)
        # pretrain key의 value가 True이면,
        if pretrain:            
            # pretrained renet_v2_50 불러오기 
            frontend_dir = os.path.join('C:\\Users\\jaemin\\data\\face\\pretrained_models', '{}.ckpt'.format(frontend))
            # slim에서 resnet_v2을 사용하기 위해서 사용하는 name scope
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                # slim에 resnet_v2_50의 net, endpoint를 들고온다.
                logits, end_points = resnet_v2.resnet_v2_50(self.X, is_training=self.is_train)
                # 불러온 pretraine resnet을 retinanet의 첫부분으로 시작한다.
                d['init_fn'] = slim.assign_from_checkpoint_fn(model_path=frontend_dir,
                                                          var_list=slim.get_model_variables(frontend))
            # resnet_v2_50의 1,2,3,4 블럭들중 4,2,1을 가져온다.
            convs = [end_points[frontend + '/block{}'.format(x)] for x in [4, 2, 1]]
        else:
            #TODO build convNet  아니면 에러
            raise NotImplementedError("Build own convNet!")

        # layer5은 block4를 받는 연산덩어리 입니다.
        with tf.variable_scope('layer5'):
            d['s_5'] = conv_layer(convs[0], 256, (1, 1), (1, 1))
            # cls는 class로 추정되고 loc는 location으로 추정됩니다.
            # 그래서 cls는 arguments중 num_class 라는 값이 있고
            # loc은 없는 형태로 num_anchors(Bbox 후보들) 만 있는 형태 같습니다.
            d['cls_head5'] = build_head_cls(d['s_5'], num_anchors, num_classes + 1)
            d['loc_head5'] = build_head_loc(d['s_5'], num_anchors)
            # 앞에 cls, loc를 reshape해서 각 layer에서 나오는 output의 형태를 다 동일하게 만들어줍니다.
            d['flat_cls_head5'] = tf.reshape(d['cls_head5'], (tf.shape(d['cls_head5'])[0], -1, num_classes + 1))
            d['flat_loc_head5'] = tf.reshape(d['loc_head5'], (tf.shape(d['loc_head5'])[0], -1, 4))

        # layer6은 layer5를 받는 연산덩어리 입니다.
        with tf.variable_scope('layer6'):
            d['s_6'] = conv_layer(d['s_5'], 256, (3, 3), (2, 2))
            d['cls_head6'] = build_head_cls(d['s_6'], num_anchors, num_classes + 1)
            d['loc_head6'] = build_head_loc(d['s_6'], num_anchors)
            d['flat_cls_head6'] = tf.reshape(d['cls_head6'], (tf.shape(d['cls_head6'])[0], -1, num_classes + 1))
            d['flat_loc_head6'] = tf.reshape(d['loc_head6'], (tf.shape(d['loc_head6'])[0], -1, 4))

        # layer7은 layer6을 받는 연산덩어리 입니다.
        with tf.variable_scope('layer7'):
            d['s_7'] = conv_layer(tf.nn.relu(d['s_6']), 256, (3, 3), (2, 2))
            d['cls_head7'] = build_head_cls(d['s_7'], num_anchors, num_classes + 1)
            d['loc_head7'] = build_head_loc(d['s_7'], num_anchors)
            d['flat_cls_head7'] = tf.reshape(d['cls_head7'], (tf.shape(d['cls_head7'])[0], -1, num_classes + 1))
            d['flat_loc_head7'] = tf.reshape(d['loc_head7'], (tf.shape(d['loc_head7'])[0], -1, 4))

        # layer4은 layer5을 resnet_block2 형태로 바꾼뒤 resnet_block2와 연산하는 덩어리 입니다.
        with tf.variable_scope('layer4'):
            d['up4'] = resize_to_target(d['s_5'], convs[1])
            d['s_4'] = conv_layer(convs[1], 256, (1, 1), (1, 1)) + d['up4']
            d['cls_head4'] = build_head_cls(d['s_4'], num_anchors, num_classes + 1)
            d['loc_head4'] = build_head_loc(d['s_4'], num_anchors)
            d['flat_cls_head4'] = tf.reshape(d['cls_head4'], (tf.shape(d['cls_head4'])[0], -1, num_classes + 1))
            d['flat_loc_head4'] = tf.reshape(d['loc_head4'], (tf.shape(d['loc_head4'])[0], -1, 4))

        # layer3은 layer4을 resnet_block1 형태로 바꾼뒤 resnet_block1와 연산하는 덩어리 입니다.
        with tf.variable_scope('layer3'):
            d['up3'] = resize_to_target(d['s_4'], convs[2])
            d['s_3'] = conv_layer(convs[2], 256, (1, 1), (1, 1)) + d['up3']
            d['cls_head3'] = build_head_cls(d['s_3'], num_anchors, num_classes + 1)
            d['loc_head3'] = build_head_loc(d['s_3'], num_anchors)
            d['flat_cls_head3'] = tf.reshape(d['cls_head3'], (tf.shape(d['cls_head3'])[0], -1, num_classes + 1))
            d['flat_loc_head3'] = tf.reshape(d['loc_head3'], (tf.shape(d['loc_head3'])[0], -1, 4))

        # 연산을 마친 head들을 flat_xx 라는 이름들로 레이어 인덱스를 붙여 적절하게 reshape해준다음 전부
        # concat으로 각각 이어줍니다.
        with tf.variable_scope('head'):
            d['cls_head'] = tf.concat((d['flat_cls_head3'],
                                       d['flat_cls_head4'],
                                       d['flat_cls_head5'],
                                       d['flat_cls_head6'],
                                       d['flat_cls_head7']), axis=1)

            d['loc_head'] = tf.concat((d['flat_loc_head3'],
                                       d['flat_loc_head4'],
                                       d['flat_loc_head5'],
                                       d['flat_loc_head6'],
                                       d['flat_loc_head7']), axis=1)

            # logits은 loc_head와 cls_head를 합쳐 축 두개 기준으로 나열한 것입니다.
            d['logits'] = tf.concat((d['loc_head'], d['cls_head']), axis=2)
            # pred는 loc_head와 softmax한 cls_head를 축 두개 기준으로 나열한 것입니다.
            d['pred'] = tf.concat((d['loc_head'], tf.nn.softmax(d['cls_head'], axis=-1)), axis=2)
        
        # 각 name scope으로 구성된 네트워크를 리턴합니다.
        return d

    # build_loss 는 total_loss 를 계산하는 메쏘드
    def _build_loss(self, **kwargs):
        #계산
        r_alpha = kwargs.pop('r_alpha', 1)
        with tf.variable_scope('losses'):
            conf_loss = focal_loss(self.logits, self.y)
            regress_loss = smooth_l1_loss(self.logits, self.y)
            total_loss = conf_loss + r_alpha * regress_loss

        # for debug
        self.conf_loss = conf_loss
        self.regress_loss = regress_loss
        return total_loss

    # def _build_pred_y(self, **kwargs):
    #     regressions  = self.pred[:, :, :4]
    #     regressions = tf.py_func(bbox_transform_inv, [self.anchors, regressions], tf.float32)
    #     pred_y = tf.concat((regressions, self.pred[:, :, 4:]), axis=2)
    #     return pred_y