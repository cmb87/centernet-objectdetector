import tensorflow as tf


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