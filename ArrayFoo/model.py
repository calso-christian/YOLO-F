from IPython.display import set_matplotlib_formats
import os
import psutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import gc
import xml.etree.ElementTree as ET
import albumentations as A
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.regularizers import l2
from keras.layers import (
    Concatenate,
    Conv2D,
    SeparableConv2D,
    Input,
    MaxPool2D,
    UpSampling2D,
    BatchNormalization)
print(tf.__version__)

architecture = {
    'backbone': 'EfficientNetV2B1',
    'SPPF_C': 480,
    'Neck_C': [120, 240],
    'Det_Blocks': 2,
    'Head_Blocks': 1,
    'Head_C': 128
}

config = {
    'LABELS': ['W', 'C'],
    'NUM_classes': 2,
    'INPUT_shape': [768, 768, 3],
    'ANCHORS':  tf.constant([
        [(110.69006964, 115.58234666), (150.43162309, 155.19736292)],
        [(68.92051207, 71.54741845), (85.42938281, 91.21690259)],
        [(38.6832461, 39.43056285), (53.86257291, 55.85377213)]],
        tf.float32) / 1300,
    'ANCHORS_shape': [3, 2],
    'PARAMS_GS_alpha': [1.5, 1.75, 2.0],
    'PARAMS_WH_power': [6.0, 4.0, 2.0],
    'PARAMS_head_scale': [1.0, 1.5, 3.5],
    'PARAMS_conf_scale': [0.02, 0.02, 0.02],
    'RES_variations': [1088, 1024, 896, 832, 768, 704, 640]
}

directory = {
    'TRAIN_Annotations': '/kaggle/input/w-and-c-hand-signals-arrayfoo/Train/Annotations/',
    'TRAIN_Images': '/kaggle/input/w-and-c-hand-signals-arrayfoo/Train/Images/',
    'VALIDATION_Annotations': '/kaggle/input/w-and-c-hand-signals-arrayfoo/Validation/Annotations/',
    'VALIDATION_Images': '/kaggle/input/w-and-c-hand-signals-arrayfoo/Validation/Images/',
    'TEST_Annotations': '/kaggle/input/w-and-c-hand-signals-arrayfoo/Test/Annotations/',
    'TEST_Images': '/kaggle/input/w-and-c-hand-signals-arrayfoo/Test/Images/',
}

colors = ['#FFA500', '#6A5ACD']


def XYXY_to_YXYX(x):
    x1, y1, x2, y2 = tf.split(x[..., :4], (1, 1, 1, 1), axis=-1)
    return tf.concat([y1, x1, y2, x2], -1)


def draw_predictions(image, pred_size, boxes, scores, labels, name=None):
    col = [[60, 20, 220], [205, 90, 105]]
    image = tf.cast(tf.squeeze(image), tf.uint8)
    idx_non_zero = tf.where(scores)
    if idx_non_zero.shape[0] == 0:
        return image, None

    img_dims = tf.cast(tf.shape(image), tf.float32)
    '''
    ratio = img_dims[1] / img_dims[0]

    if ratio > 0:
        offset = tf.tile(
            tf.stack([
                0.,
                (pred_size - pred_size / ratio) / 2.]),
            [2])[tf.newaxis, ...]
    elif ratio < 0:
        offset = tf.tile(
            tf.stack([
                (pred_size - pred_size / ratio) * 2.,
                0.]),
            [2])[tf.newaxis, ...]
    else:
        offset = tf.constant([0., 0., 0., 0.])

    pad = tf.stack([
        img_dims[1] + offset[0, 0] - pred_size,
        img_dims[0] + offset[0, 1] - pred_size])

    pad = tf.tile(pad, [2])[tf.newaxis, ...] '''
    boxes = tf.cast((tf.gather_nd(boxes, idx_non_zero)) *
                    pred_size, tf.int32)  # + offset - pad

    scores = tf.gather_nd(scores, idx_non_zero)
    labels = tf.gather_nd(labels, idx_non_zero)
    zeros = tf.zeros(image.shape)
    predictions = []
    for CLS in tf.range(config['NUM_classes']):
        idx_CLS = tf.where(tf.cast(labels, tf.int32) == CLS)
        b = tf.gather_nd(boxes, idx_CLS).numpy()
        for i in tf.range(b.shape[0]):
            start, end = b[i, :2], b[i, 2:]
            mask = tf.cast(cv.rectangle(
                zeros.numpy(), start, end, col[CLS], 4), tf.uint8)
            image = image * tf.cast((mask == 0), tf.uint8) + mask
            predictions.append(
                (CLS, b[i])
            )

    return image, predictions


def Conv_SiLU(x, filters, kernel_size, strides=1, depth_multiplier=1, mode='Conv2D', batch_norm=True):
    if mode == 'SeparableConv2D':
        x = SeparableConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            depth_multiplier=depth_multiplier,
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(),
            use_bias=not batch_norm
        )(x)
    else:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=not batch_norm,
                   kernel_regularizer=l2(0.0005),
                   kernel_initializer='he_normal')(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = tf.nn.silu(x)
    return x


def Conv_MOD(x, filters):
    x = Conv_SiLU(x, filters, 1)
    x = Conv_SiLU(x, filters, 3)
    return x


class PAN:
    def __init__(self, Conv):
        self.Conv = Conv

    def Conv_Block(self, x, filters):
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.nn.silu(x)
        x = self.Conv(x, filters, 1)
        return x

    def Detection_Block(self, x, C_in, C_out=None):
        if C_out == None:
            C_out = C_in
        x_1 = self.Conv(x, C_in / 2, 1)
        x_2 = self.Conv(x, C_in / 2, 1)
        for _ in range(architecture['Det_Blocks']):
            x_2 = self.Conv_Block(x_2, C_in / 2)
        x = Concatenate()([x_1, x_2])
        x = self.Conv(x, C_out, 1)
        return x

    def SPPF(self, t, filters, pool_size=5, name='SPPF'):
        pool = MaxPool2D(pool_size, 1, padding='same')
        x = inputs = Input(t.shape[1:])
        x = self.Conv(x, filters // 2, 1)
        p_1 = pool(x)
        p_2 = pool(p_1)
        p_3 = pool(p_2)
        x = tf.concat([x, p_1, p_2, p_3], axis=-1)
        x = self.Conv(x, filters, 1)
        return Model(inputs, x, name=name)(t)

    def FPN_head(self, t, filters, conv=True, name=None):
        if isinstance(t, tuple):
            inputs = Input(t[0].shape[1:]), Input(t[1].shape[1:])
            x, x_skip = inputs
            x = UpSampling2D(2, interpolation='bilinear')(x)
            x = Concatenate()([x, x_skip])
            x = self.Detection_Block(x, filters)
            if conv:
                x = self.Conv(x, filters // 2, 1)
        else:
            x = inputs = Input(t.shape[1:])
            x = self.Conv(x, filters, 1)
        return Model(inputs, x, name=name)(t)

    def PAN_head(self, t, filters, name=None):
        inputs = Input(t[0].shape[1:]), Input(t[1].shape[1:])
        x, y = inputs
        x = self.Conv(x, filters, 3, 2)
        x = Concatenate()([x, y])
        x = self.Detection_Block(x, C_in=filters, C_out=filters*2)
        return Model(inputs, x, name=name)(t)

    def __call__(self, x_in):
        x_1, x_2, x_3 = x_in
        x_3 = self.SPPF(x_3, architecture['SPPF_C'])
        x_3 = y_3 = self.FPN_head(
            x_3, architecture['Neck_C'][1], name='FPN_13')
        x_3 = y_2 = self.FPN_head(
            (x_3, x_2), architecture['Neck_C'][1], name='FPN_26')
        x_3 = y_1 = self.FPN_head(
            (x_3, x_1), architecture['Neck_C'][0], conv=False, name='PAN_52')
        y_2 = self.PAN_head(
            (y_1, y_2), architecture['Neck_C'][0], name='PAN_26')
        y_3 = self.PAN_head(
            (y_2, y_3), architecture['Neck_C'][1], name='PAN_13')
        return (y_1, y_2, y_3)


def Decoupled_Head(filters, NUM_anchors, NUM_classes, name=None):
    class Head_Transformation(keras.layers.Layer):
        def __init__(self, NUM_anchors, NUM_classes):
            super().__init__()
            self.NUM_anchors = NUM_anchors
            self.NUM_classes = NUM_classes

        def call(self, x):
            return tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], self.NUM_anchors, self.NUM_classes + 5))

    def builder(t):
        x = inputs = Input(t.shape[1:])
        x_1 = x_2 = Conv_SiLU(x, architecture['Head_C'], 1)
        for _ in range(architecture['Head_Blocks']):
            x_1 = Conv_SiLU(x_1, architecture['Head_C'], 3)
            x_2 = Conv_SiLU(x_2, architecture['Head_C'], 3)
        x_Cls = Conv_SiLU(x_1, NUM_anchors * (NUM_classes),
                          1, batch_norm=False)
        x_Box = Conv_SiLU(x_2, NUM_anchors * 4, 1, batch_norm=False)
        x_Conf = Conv_SiLU(x_2, NUM_anchors * 1, 1, batch_norm=False)
        x = tf.concat([x_Box, x_Conf, x_Cls], axis=-1)
        x = Head_Transformation(NUM_anchors, NUM_classes)(x)
        return tf.keras.Model(inputs, x, name=name)(t)
    return builder


class Output_Activation(tf.keras.layers.Layer):
    def __init__(self, NUM_classes, anchors, GS_alpha, WH_power, training=False):
        super().__init__()
        self.training = training
        self.NUM_classes = NUM_classes
        self.anchors = anchors
        self.GS_alpha = GS_alpha
        self.WH_power = WH_power

    def call(self, x):
        grid_size = tf.shape(x)[1:3]
        box_xy, box_wh, confidence, class_probs = tf.split(
            x, (2, 2, 1, self.NUM_classes), axis=-1)
        box_xy = tf.sigmoid(box_xy)
        box_wh = ((2 * tf.sigmoid(box_wh)) ** self.WH_power) * self.anchors

        if not self.training:
            confidence = tf.sigmoid(confidence)
            class_probs = tf.sigmoid(class_probs)

        grid = tf.meshgrid(tf.range(grid_size[0]), tf.range(grid_size[1]))
        grid = tf.stack(grid, axis=-1)[..., tf.newaxis, :]
        box_xy = (self.GS_alpha * box_xy - (self.GS_alpha - 1) / 2 +
                  tf.cast(grid, x.dtype)) / tf.cast(grid_size, x.dtype)
        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
        return bbox, confidence, class_probs


class EfficientNetV2:
  def __init__(self, mode='EfficientNetV2S', trainable=True):
    self.config = ['block2c_add', 'block4a_expand_activation', 'block6a_expand_activation']
    if mode == 'EfficientNetV2B0': 
        self.model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0
        self.config = ['block2b_add', 'block4a_expand_activation', 'block6a_expand_activation']
    elif mode == 'EfficientNetV2B1': self.model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1
    elif mode == 'EfficientNetV2B2': self.model = tf.keras.applications.efficientnet_v2.EfficientNetV2B2
    elif mode == 'EfficientNetV2B3': self.model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3
    else:
      mode = 'EfficientNetV2S'
      self.model = tf.keras.applications.efficientnet_v2.EfficientNetV2S
    
    self.model = self.model(include_top=False,
                            weights=None,
                            input_tensor=None,
                            input_shape=None,
                            include_preprocessing=False,
                            classifier_activation=None)
    self.mode = mode
    self.trainable = trainable
  
  def __call__(self):
    x = inputs = Input([None, None, 3])
    m = Model(inputs=self.model.inputs, outputs=self.model.get_layer(self.config[0]).output)
    n = Model(inputs=m.inputs, outputs = self.model.get_layer(self.config[1]).output)
    o = Model(inputs=n.inputs, outputs=(m.output, n.output, self.model.get_layer(self.config[2]).output))
    x = o(x)
    model = Model(inputs=inputs, outputs=x, name=self.mode)
    model.trainable = self.trainable
    return model


class Post_Process(keras.layers.Layer):
    def __init__(self, NUM_classes, max_boxes=25, IoU_thresh=0.5, score_thresh=0.55):
        super().__init__()
        self.NUM_classes = NUM_classes
        self.max_boxes = max_boxes
        self.IoU_thresh = IoU_thresh
        self.score_thresh = score_thresh

    def NMS(self, outputs):
        bbox, confidence, class_probs = [], [], []
        for o in outputs:
            bbox.append(tf.reshape(
                o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
            confidence.append(tf.reshape(
                o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
            class_probs.append(tf.reshape(
                o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

        bbox = tf.maximum(tf.concat(bbox, axis=1), 0.)[..., tf.newaxis, :]
        confidence = tf.concat(confidence, axis=1)
        class_probs = tf.concat(class_probs, axis=1)

        if self.NUM_classes == 1:
            scores = confidence
        else:
            scores = confidence * class_probs

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.cast(bbox, tf.float32),
            scores=tf.cast(scores, tf.float32),
            max_output_size_per_class=self.max_boxes,
            max_total_size=self.max_boxes,
            iou_threshold=self.IoU_thresh,
            score_threshold=self.score_thresh)

        return boxes, scores, classes, valid_detections

    def call(self, x):
        x = self.NMS(x)
        return x


def YOLOv3_MOD(Backbone, Neck, Head, SHAPE_input, NUM_anchors, NUM_classes, training=True):
    x = inputs = Input(SHAPE_input, name='input')
    if not training:
        x = tf.image.resize_with_pad(x, SHAPE_input[0], SHAPE_input[1])
    x = tf.keras.layers.Rescaling(scale=1./255.)(x)
    x = Backbone()(x)
    y_1, y_2, y_3 = Neck(x)
    y_3 = Head(240, NUM_anchors, NUM_classes, 'Head_13')(y_3)
    y_2 = Head(120, NUM_anchors, NUM_classes, 'Head_26')(y_2)
    y_1 = Head(60, NUM_anchors, NUM_classes, 'Head_52')(y_1)
    y = [y_3, y_2, y_1]
    if training:
        return Model(inputs, y, name='YOLOv3_MOD')
    else:
        y = [Output_Activation(config['NUM_classes'],
                               config['ANCHORS'][level],
                               config['PARAMS_GS_alpha'][level],
                               config['PARAMS_WH_power'][level],
                               training=False)(y[level]) for level in [0, 1, 2]]
        y = Post_Process(NUM_classes, 20, 0.35, 0.75)(y)

    return Model(inputs, y, name='YOLOv3_MOD')
'''
architecture = {
  'backbone': 'EfficientNetV2B0',
  'SPPF_C': 256,
  'Neck_C': [64, 128],
  'Det_Blocks': 1,
  'Head_Blocks': 1,
  'Head_C': 128
}
'''
structure = [EfficientNetV2(mode=architecture['backbone'], trainable=True), PAN(
    Conv_SiLU), Decoupled_Head]
model = YOLOv3_MOD(*structure, config['INPUT_shape'],
                   config['ANCHORS_shape'][1], config['NUM_classes'], training=False)
model.load_weights(
    r"ArrayFoo\weights\A\A_10.h5")


# NEW MODELS

# N_1
# A_10, A_11
