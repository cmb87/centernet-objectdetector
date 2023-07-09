from keras_resnet import models as resnet_models
from keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, Dropout, Add
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.initializers import normal, constant, zeros
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf

# https://github.com/xuannianz/keras-CenterNet/blob/master/models/resnet.py

def centernet(num_classes, backbone='resnet50', input_size=512, freeze_bn=False):

    
    assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

    image_input = Input(shape=(input_size, input_size, 3))

    #i = ByteLayer()(image_input) # 0-1 => 0-255

    if backbone == 'resnet18':
        resnet = resnet_models.ResNet18(image_input, include_top=False, freeze_bn=freeze_bn)
    elif backbone == 'resnet34':
        resnet = resnet_models.ResNet34(image_input, include_top=False, freeze_bn=freeze_bn)
    elif backbone == 'resnet50':
        resnet = resnet_models.ResNet50(image_input, include_top=False, freeze_bn=freeze_bn)
        # resnet = ResNet50(input_tensor=image_input, include_top=False)
    elif backbone == 'resnet101':
        resnet = resnet_models.ResNet101(image_input, include_top=False, freeze_bn=freeze_bn)
    else:
        resnet = resnet_models.ResNet152(image_input, include_top=False, freeze_bn=freeze_bn)

    # (b, 16, 16, 2048)

    for n,l in enumerate(resnet.layers):
        print(n, l.name, l.get_output_at(0).get_shape().as_list())

    C2 = resnet.layers[23].output
    C3 = resnet.layers[43].output
    C4 = resnet.layers[63].output
    C5 = resnet.outputs[-1]

    x2 = Dropout(rate=0.060)(C2)
    x3 = Dropout(rate=0.125)(C3)
    x4 = Dropout(rate=0.250)(C4)
    x5 = Dropout(rate=0.500)(C5)

    x = x5

    num_filters = 256


    for i,y in enumerate([x4,x3,x2]):
        num_filters = num_filters // pow(2, i)
        # x = Conv2D(num_filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(
        #     x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        x = Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        
        y = Conv2D(num_filters, (1, 1), padding='same')(y)
        x = Add()([x,y])
        x = BatchNormalization()(x)
        x = ReLU()(x)


    model = Model(inputs=image_input, outputs=x)

    # # hm header
    # y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    # y1 = BatchNormalization()(y1)
    # y1 = ReLU()(y1)
    # y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # # wh header
    # y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    # y2 = BatchNormalization()(y2)
    # y2 = ReLU()(y2)
    # y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # # reg header
    # y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    # y3 = BatchNormalization()(y3)
    # y3 = ReLU()(y3)
    # y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

    # loss_ = Lambda(loss, name='centernet_loss')(
    #     [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    # model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    # # detections = decode(y1, y2, y3)
    # detections = Lambda(lambda x: decode(*x,
    #                                      max_objects=max_objects,
    #                                      score_threshold=score_threshold,
    #                                      nms=nms,
    #                                      flip_test=flip_test,
    #                                      num_classes=num_classes))([y1, y2, y3])
    # prediction_model = Model(inputs=image_input, outputs=detections)


    return model

if (__name__ == "__main__"):

    model = centernet(num_classes=4, backbone='resnet18', input_size=512, freeze_bn=True)
    print(model.summary())