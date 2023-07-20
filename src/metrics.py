import tensorflow as tf
from tensorflow import keras
from data.labels import decode



def _getBoxes(bc, wh):
    # Convert to dimensions

    # According to official squeezeDet paper
    # bxy = self.axycr + self.awhr * y[:,:, 1:3]

    # Alternative

    x1 = tf.expand_dims(bc[:, :, 0] - 0.5 * wh[:, :, 0], -1)
    y1 = tf.expand_dims(bc[:, :, 1] - 0.5 * wh[:, :, 1], -1)
    x2 = tf.expand_dims(bc[:, :, 0] + 0.5 * wh[:, :, 0], -1)
    y2 = tf.expand_dims(bc[:, :, 1] + 0.5 * wh[:, :, 1], -1)

    # Convert from x,y,w,h to x1,y1,x2,y2
    return x1, y1, x2, y2



def _getIou(bcTrue, whTrue, bcPred, whPred):
    [x1a, y1a, x2a, y2a] = _getBoxes(bcTrue, whTrue)
    [x1b, y1b, x2b, y2b] = _getBoxes(bcPred, whPred)

    dx = tf.math.maximum(tf.math.minimum(x2a,x2b) - tf.math.maximum(x1a,x1b), tf.zeros_like(x1a))
    dy = tf.math.maximum(tf.math.minimum(y2a,y2b) - tf.math.maximum(y1a,y1b), tf.zeros_like(y1a))

    intersection = dx*dy
    area1 = (x2a - x1a) * (y2a - y1a)
    area2 = (x2b - x1b) * (y2b - y1b)
    iou = intersection / (area1 + area2 - intersection + 1e-6)

    return tf.squeeze(iou, -1)




class IoU(keras.metrics.Metric):
    def __init__(self, name = 'IoU', **kwargs):
        super(IoU, self).__init__(**kwargs)

        self.iou = self.add_weight('iou', initializer = 'zeros')


    def update_state(self, y_true, y_pred,sample_weight=None):

        scoreTrue, classesTrue, bcTrue, whTrue = decode(y_true)
        scorePred, classesPred, bcPred, whPred = decode(y_pred)


        iou = _getIou(bcTrue, whTrue, bcPred, whPred)

        self.iou.assign_add(tf.reduce_mean(tf.cast(iou, self.dtype)))




        # Old but gold
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.math.greater(y_pred, 0.5)      
        true_p = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_p = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        
     #   self.fp.assign_add(tf.reduce_sum(tf.cast(false_p, self.dtype)))



    def reset_state(self):
        self.iou.assign(0)


    def result(self):
        return self.iou