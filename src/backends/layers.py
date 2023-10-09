import tensorflow as tf


class WeightedAddLayer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(WeightedAddLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.B, self.H, self.W = input_shape[0][0], input_shape[0][1], input_shape[0][2]
        self.nfeat = input_shape[0][3]

        self.concat = tf.keras.layers.Concatenate()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(units=4, activation="relu") # bottleneck==2
        self.dense2 = tf.keras.layers.Dense(units=self.nfeat, activation="sigmoid")
        self.multi = tf.keras.layers.Multiply()


    def call(self, x, training=False):
        """
        Takes as input normalized data [0.0-1.0] and transforms them to Cafe normalization
        :type training: object
        """

        w = self.concat(x)
        w = self.pool(w)
        w = self.dense1(w)
        w = self.dense2(w) # [B,C]

        return self.multi([w,x[0]]) + self.multi([(1.0-w),x[1]])


class ChannelAttentionLayer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ChannelAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.B, self.H, self.W = input_shape[0], input_shape[1], input_shape[2]
        self.nfeat = input_shape[3]

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(units=2, activation="relu") # bottleneck==2
        self.dense2 = tf.keras.layers.Dense(units=self.nfeat, activation="sigmoid")
        self.multi = tf.keras.layers.Multiply()


    def call(self, x, training=False):
        """
        Takes as input normalized data [0.0-1.0] and transforms them to Cafe normalization
        :type training: object
        """
        w = self.pool(x)
        w = self.dense1(w)
        w = self.dense2(w) # [B,C]

        return self.multi([w,x])


class SpatialAttentionLayer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.B, self.H, self.W = input_shape[0], input_shape[1], input_shape[2]
        self.nfeat = input_shape[3]

        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(self.nfeat, (1, 1), activation='linear',padding='same')
        self.act = tf.keras.layers.Activation(tf.math.sigmoid)
        self.multi = tf.keras.layers.Multiply()

    def call(self, x, training=False):
        """
        Takes as input normalized data [0.0-1.0] and transforms them to Cafe normalization
        :type training: object
        """
        w = self.conv(x)
        w = self.bn(w)
        w = self.act(w)

        return self.multi([w,x])






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
    



class FireModuleNew(tf.keras.Model):
    def __init__(self, s1, e1, e3, mobile=False, dilation=(1,1), **kwargs):
        super(FireModuleNew, self).__init__(**kwargs)
        self.s1 = s1
        self.e1 = e1
        self.e3 = e3
        self.dilation = dilation
        self.mobile= mobile

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
            self.s1, (1, 1), activation='linear',
            padding='same', dilation_rate=self.dilation,
            kernel_initializer=kinit, bias_initializer=binit,
        )
        self.e1 = tf.keras.layers.Conv2D(
            self.e1, (1, 1), activation='linear',
            padding='same', dilation_rate=self.dilation,
            kernel_initializer=kinit, bias_initializer=binit,
        )
        if self.mobile:
            self.e3 = tf.keras.layers.DepthwiseConv2D(kernel_size = (3,3), padding = 'same', dilation_rate=self.dilation)
        else:
            self.e3 = tf.keras.layers.Conv2D(
                self.e3, (3, 3), activation='linear',
                padding='same', dilation_rate=self.dilation,
                kernel_initializer=kinit, bias_initializer=binit,
            )

        #self.cat = tf.keras.layers.Concatenate()
        self.cat = tf.keras.layers.Add()

        self.drop1 = tf.keras.layers.Dropout(0.2)
        self.drop2 = tf.keras.layers.Dropout(0.2)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
 

        self.relu1 = tf.keras.layers.ReLU()
        self.relu2 =  tf.keras.layers.ReLU()


    def call(self, input_tensor, training=False):

        x = self.s1(input_tensor, training=training)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x1 = self.e1(x, training=training)
        x1 = self.drop1(x1, training=training)

        x3 = self.e3(x, training=training)
        x3 = self.drop2(x3, training=training)

        x = self.cat([x1,x3])
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        return x
    


    
class Residual(tf.keras.Model):
    def __init__(self, nf, dilation=(1,1), strides=1, dropout=0.2, **kwargs):
        super(Residual, self).__init__(**kwargs)

        self.nf = nf
        self.dilation = dilation
        self.dropout = dropout

    def build(self, input_shape):
        
        nf0 = input_shape[3]


        self.conv1 = tf.keras.layers.Conv2D(int(0.5*self.nf), (1, 1) , padding='same', dilation_rate=self.dilation, use_bias=True)
        self.conv2 = tf.keras.layers.Conv2D(int(0.5*self.nf), (3, 3), padding='same', dilation_rate=self.dilation, use_bias=True)
        self.conv3 = tf.keras.layers.Conv2D(self.nf, (1, 1), padding='same', dilation_rate=self.dilation, use_bias=True)


        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.elu1 = tf.keras.layers.ELU(alpha=1.0)
        self.elu2 = tf.keras.layers.ELU(alpha=1.0)
        self.elu3 = tf.keras.layers.ELU(alpha=1.0)

        self.drop1 = tf.keras.layers.Dropout(self.dropout)
        self.drop2 = tf.keras.layers.Dropout(self.dropout)
        self.drop3 = tf.keras.layers.Dropout(self.dropout)


        if nf0 == self.nf:
            self.need_skip = False
            self.skipConv = None
            print("None...")
        else:
            self.need_skip = True
            self.skipConv = tf.keras.layers.Conv2D(self.nf, (1, 1), activation='relu', padding='same', dilation_rate=self.dilation)
            print(f"Inputshape {nf0} Outputshape {self.nf} ({input_shape})")
        

    def call(self, input_tensor, training=None):

        xred = input_tensor if not self.need_skip else self.skipConv(input_tensor)

        
        x = self.conv1(input_tensor, training=training)
        x = self.bn1(x, training=training)
        x = self.elu1(x, training=training)
      #  x = self.drop1(x, training=training)

        x = self.conv2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.elu2(x, training=training)
      #  x = self.drop2(x, training=training)

        x = self.conv3(x, training=training)
        x = self.bn3(x, training=training)
        x = self.elu3(x, training=training)
     #   x = self.drop3(x, training=training)

        x = x + xred
        return x



class HourglassModule(tf.keras.Model):
    def __init__(self, nfilters, useLight=False, useFire=True, **kwargs):
        super(HourglassModule, self).__init__(**kwargs)

        nf = nfilters[0]
        ndepths = len(nfilters)

        if useFire:
            self.low1 = FireModuleNew(nf//2, nf, nf, mobile=useLight)
            self.low3 = FireModuleNew(nf//2, nf, nf, mobile=useLight)
            self.up1 = FireModuleNew(nf//2, nf, nf, mobile=useLight)

        self.up1_ca = ChannelAttentionLayer()
        self.up1_sa = SpatialAttentionLayer()

        self.hg_ca = ChannelAttentionLayer()
        self.hg_sa = SpatialAttentionLayer()
        self.add = WeightedAddLayer() #tf.keras.layers.Add()


        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        self.upsample = tf.keras.layers.UpSampling2D(2, interpolation="bilinear")

        if ndepths>1:
            self.hg = HourglassModule(nfilters=nfilters[1:])
        else:
            if useFire:
                self.hg = FireModuleNew(nf//2, nf, nf, mobile=useLight)


    def call(self, x, training=None):

        xup = self.up1_ca(x)
        xup = self.up1_sa(xup)
        xup  = self.up1(xup, training=training)


        xlo = self.pool1(x, training=training)
        xlo = self.low1(xlo, training=training)
        xlo = self.hg_ca(xlo)
        xlo = self.hg_sa(xlo)
        xlo = self.hg(xlo, training=training)
        xlo = self.upsample(xlo, training=training)
        xlo = self.low3(xlo, training=training)

        x = self.add([xup,xlo])
        

        return x



def getHourglass(iw, ih, ic, nfeat=128, nfilters=[128,128,256,256,256]):

    i = tf.keras.layers.Input((ih,iw,ic), name="rgb")


  #  # ========= Entry Layers =================
    x = tf.keras.layers.Conv2D(nfeat//2, (3,3), name="entry00", padding="same", dilation_rate=2, activation="relu")(i) #

    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Residual(nfeat//2, name="entry01")(x)
    x = Residual(nfeat//2, name="entry02")(x)
    
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Residual(nfeat, name="entry03")(x)

    # ========= First Hourglas =================
    y = HourglassModule(nfilters=nfilters,  name="hourglass1", useFire=True)(x)

    model = tf.keras.Model(i,y)
    return model

if __name__ == "__main__":


    model = getHourglass(512, 512, 3, nfeat=128, nfilters=[128,128,256,256,512])


    print(model.summary())