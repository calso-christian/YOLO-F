from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import tensorflow as tf
from tensorflow import keras

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

    boxes = tf.cast((tf.gather_nd(boxes, idx_non_zero)) * pred_size, tf.int32)

    scores = tf.gather_nd(scores, idx_non_zero)
    labels = tf.gather_nd(labels, idx_non_zero)
    zeros = tf.zeros(image.shape)
    predictions = []
    for CLS in tf.range(config['NUM_classes']):
        idx_CLS = tf.where(tf.cast(labels, tf.int32) == CLS)
        b = tf.gather_nd(boxes, idx_CLS).numpy()
        for i in tf.range(b.shape[0]):
            start, end = b[i, :2], b[i, 2:]
            mask = tf.cast(cv.rectangle(zeros.numpy(), start, end, col[CLS], 4), tf.uint8)
            image = image * tf.cast((mask == 0), tf.uint8) + mask
            predictions.append((CLS, b[i]))

    return image, predictions


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


@tf.function
def predict(model, x):
  y = model(x, training=False)
  y = [Output_Activation(config['NUM_classes'],
                        config['ANCHORS'][level],
                        config['PARAMS_GS_alpha'][level],
                        config['PARAMS_WH_power'][level],
                        training=False)(y[level]) for level in [0, 1, 2]]
  y = Post_Process(config['NUM_classes'], 15, 0.25, 0.6)(y)
  return y

model = tf.keras.models.load_model(r'ArrayFoo\weights\N\N_1')