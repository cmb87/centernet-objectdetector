import tensorflow as tf

def se_unit(x, bottleneck=2):
    with tf.variable_scope(None, 'SE_module'):
        n, h, w, c = x.get_shape().as_list()

        kernel_size = resolve_shape(x)
        x_pool = slim.avg_pool2d(x, kernel_size, stride=1)
        x_pool = tf.reshape(x_pool, shape=[-1, c])
        fc = slim.fully_connected(x_pool, bottleneck, activation_fn=tf.nn.relu,
                                  biases_initializer=None)
        fc = slim.fully_connected(fc, c, activation_fn=tf.nn.sigmoid,
                                  biases_initializer=None)
        if n is None:
            channel_w = tf.reshape(fc, shape=tf.convert_to_tensor([tf.shape(x)[0], 1, 1, c]))
        else:
            channel_w = tf.reshape(fc, shape=[n, 1, 1, c])

        x = tf.multiply(x, channel_w)
    return x

def sa_unit(x):
    with tf.variable_scope(None, 'SA_module'):
        shape=x.get_shape().as_list()
        y=tf.keras.layers.conv2d(x,shape[-1],kernel_size=1,stride=1,biases_initializer=None,activation_fn=None)
        y=tf.keras.layers.batch_norm(y,activation_fn=None, fused=False)
        y=tf.nn.sigmoid(y)
        x=tf.multiply(x,y)
        return x



class NormalizationLayer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NormalizationLayer, self).__init__(**kwargs)

    def get_config(self):
        return {}
    #
    def call(self, x, training=False):
        """
        Takes as input normalized data [0.0-1.0] and transforms them to Cafe normalization
        :type training: object
        """
        mean = tf.expand_dims( tf.expand_dims(tf.constant([0.40789655, 0.44719303, 0.47026116]),0),0) 
        std = tf.expand_dims( tf.expand_dims(tf.constant([0.2886383, 0.27408165, 0.27809834]),0),0)


        return (x - mean)/std


class ByteLayer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ByteLayer, self).__init__(**kwargs)

    def get_config(self):
        return {}
    #
    def call(self, x, training=False):
        """
        Takes as input normalized data [0.0-1.0] and transforms them to Cafe normalization
        :type training: object
        """
        return 255*x



class PostprocessingLayer(tf.keras.Model):
    def __init__(self, iw=320, ih=320, ndet=50, **kwargs):
        super(PostprocessingLayer, self).__init__(**kwargs)
        self.ndet = ndet
        self.iw = iw
        self.ih = ih

    def get_config(self):
        return {}

    def build(self, input_shape):
        self.B, self.H, self.W = input_shape[0], input_shape[1], input_shape[2]
        self.C = input_shape[3]-4
        self.scale = tf.constant([[[float(self.iw)/self.W, float(self.ih)/self.H]]], dtype=tf.float32)

    def call(self, y, training=False):
        """
        Takes as input normalized data [0.0-1.0] and transforms them to Cafe normalization
        :type training: object
        """
        B, H, W, C, K = self.B, self.H, self.W, self.C, self.ndet
        scale = self.scale

        # Split Tensors
        #hm = tf.slice(y, [0, 0, 0, 0], [-1, H, W, C])
        #wh = tf.math.exp(tf.slice(y, [0, 0, 0, C], [-1, H, W, C+2])) - 1.0
        #bc = tf.slice(y, [0, 0, 0, C+2], [-1, H, W, C+4])

        hm = y[...,:C]
        wh = tf.math.exp(y[...,C:C+2]) - 1.0
        bc = y[...,C+2:C+4]

        # Extend
        wh = tf.reshape(wh,(-1,W*H,2))
        bc = tf.reshape(bc,(-1,W*H,2))

        # NMS
        hmax = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(hm)
        mask = tf.cast(tf.equal(hm, hmax), tf.float32)
        keep =  hmax * mask # [B,H,W,C]

        # Reshape and sort
        score = tf.reshape(keep,(-1,H*W*C)) # [B,HWC]
        idx = tf.argsort(score, axis=-1, direction='DESCENDING')

        # =========================
        k = tf.math.floormod(idx, C)
        j = tf.math.floormod(tf.math.floordiv(idx, C), W)
        i = tf.math.floordiv(idx, C*W)

        #i = tf.slice(i, [0, 0], [-1, K]) 
        #j = tf.slice(j, [0, 0], [-1, K]) 
        #k = tf.slice(k, [0, 0], [-1, K]) 
        i = i[:,:K]
        j = j[:,:K]
        k = k[:,:K]


        idx2d = j + i*W


        #idx = tf.slice(idx, [0, 0], [-1, K])
        idx = idx[:,:K]

        # =========================
        classes = k # [B,K]
        score = tf.gather(params=score, indices=idx, batch_dims=1)  # [B,K]
        bc = scale*tf.cast(tf.stack((j,i),axis=-1),tf.float32) + tf.gather(params=bc, indices=idx2d, batch_dims=1) # [B,K,2]
        wh = scale*tf.gather(params=wh, indices=idx2d, batch_dims=1)  # [B,K,2]

        return score,classes,bc,wh



class FireModule(tf.keras.Model):
    def __init__(self, s1, e1, e3, dilation=(1,1), **kwargs):
        super(FireModule, self).__init__(**kwargs)
        self.s1 = s1
        self.e1 = e1
        self.e3 = e3
        self.dilation = dilation

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            's1': self.s1,
            'e1': self.e1,
            'e1': self.e3,
            'dilation': self.dilation
        })
        return config

    def build(self, input_shape):
        kinit = tf.random_normal_initializer(
            mean=0.0, stddev=0.001, seed=None
        )
        binit = tf.zeros_initializer()
        
        self.s1 = tf.keras.layers.Conv2D(
            self.s1, (1, 1), activation='relu',
            padding='same', dilation_rate=self.dilation,
            kernel_initializer=kinit, bias_initializer=binit,
        )
        self.e1 = tf.keras.layers.Conv2D(
            self.e1, (1, 1), activation='relu',
            padding='same', dilation_rate=self.dilation,
            kernel_initializer=kinit, bias_initializer=binit,
        )
        self.e3 = tf.keras.layers.Conv2D(
            self.e3, (3, 3), activation='relu',
            padding='same', dilation_rate=self.dilation,
            kernel_initializer=kinit, bias_initializer=binit,
        )
        #self.bn = BatchNormalization(momentum=0.99)
        self.cat = tf.keras.layers.Concatenate()

        self.drop1 = Dropout(0.2)
        self.drop2 = Dropout(0.2)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.s1(input_tensor, training=training)
        x = self.bn1(x, training=training)

        x1 = self.e1(x, training=training)
        x1 = self.drop1(x1, training=training)

        x3 = self.e3(x, training=training)
        x3 = self.drop1(x3, training=training)

        x = self.cat([x1,x3])
        x = self.bn2(x, training=training)


        return x