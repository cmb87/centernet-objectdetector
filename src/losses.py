import tensorflow as tf



def centerNetLoss(ytrue, ypred, alpha=2.0, beta=4.0):

    C = tf.shape(ytrue)[-1] - 4

    hmTrue, whLogTrue, pdeltaTrue = tf.split(ytrue, [C, 2, 2], axis=-1)
    hmPred, whLogPred, pdeltaPred = tf.split(ypred, [C, 2, 2], axis=-1)
    hmPred = tf.clip_by_value(hmPred, 1e-4, 1.0 - 1e-4)

    m = tf.cast(tf.greater_equal(hmTrue,1.0), tf.float32) # [B,H,W,C]
    N = tf.reduce_sum(m, axis=[1,2,3]) + 1 

    # Heatmap error
    a = tf.pow(1.0-hmPred, alpha)*tf.math.log(hmPred)
    b = tf.pow(1.0-hmTrue, beta)*tf.pow(hmPred, alpha)*tf.math.log(1.0-hmPred)

    lossHm = -tf.reduce_sum(m*a + (1.0-m)*b, axis=[1,2,3])


    # Local and box error
    mask = tf.expand_dims(tf.cast(tf.greater(pdeltaTrue[...,0],0.0), tf.float32),-1) # [B,H,W,1]
    N = tf.reduce_sum(mask, axis=[1,2,3]) + 1 

    lossWh =      tf.reduce_sum( mask*tf.math.abs(whLogPred - whLogTrue),   axis=[1,2,3])
    lossPdelta =  tf.reduce_sum( mask*tf.math.abs(pdeltaPred - pdeltaTrue), axis=[1,2,3])

    return tf.reduce_mean( 1.0/N*(lossHm + 0.1* lossWh + lossPdelta ))

