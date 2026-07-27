import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, Model

def MobileNetV3_Small_CenterNet(input_shape=(256, 256, 3), num_features=256):
    """
    Creates a MobileNetV3-Small backbone CenterNet model.
    Extracts multi-scale feature maps from intermediate layers of the pretrained 
    MobileNetV3Small network and fuses them using dynamic upsampling layers.
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. Instantiate the pretrained base model
    base_model = MobileNetV3Small(input_shape=input_shape, include_top=False, weights="imagenet")
    
    # 2. Extract multi-scale feature maps
    # Stage outputs: x4 (1/4), x8 (1/8), x16 (1/16), x32 (1/32)
    # We retrieve intermediate tensors by querying the layers by their exact names
    x4 = base_model.get_layer("expanded_conv_project_bn").output   # Shape: (None, 64, 64, 16)
    x8 = base_model.get_layer("expanded_conv_2_project_bn").output # Shape: (None, 32, 32, 24)
    x16 = base_model.get_layer("expanded_conv_7_project_bn").output # Shape: (None, 16, 16, 48)
    x32 = base_model.output                                      # Shape: (None, 8, 8, 576)

    # 3. Create neck projections to uniform dimension (num_features = 256)
    proj_x4 = layers.Conv2D(num_features, kernel_size=1, strides=1, padding="same", use_bias=False)(x4)
    proj_x4 = layers.BatchNormalization()(proj_x4)
    proj_x4 = layers.ReLU()(proj_x4)
    
    proj_x8 = layers.Conv2D(num_features, kernel_size=1, strides=1, padding="same", use_bias=False)(x8)
    proj_x8 = layers.BatchNormalization()(proj_x8)
    proj_x8 = layers.ReLU()(proj_x8)
    
    proj_x16 = layers.Conv2D(num_features, kernel_size=1, strides=1, padding="same", use_bias=False)(x16)
    proj_x16 = layers.BatchNormalization()(proj_x16)
    proj_x16 = layers.ReLU()(proj_x16)
    
    proj_x32 = layers.Conv2D(num_features, kernel_size=1, strides=1, padding="same", use_bias=False)(x32)
    proj_x32 = layers.BatchNormalization()(proj_x32)
    proj_x32 = layers.ReLU()(proj_x32)

    # 4. Fusing Head via consecutive bilinear upsampling & elements-wise summation (CenterNet Neck)
    # Step A: Upsample x32 to x16 resolution and sum
    up_x32 = layers.UpSampling2D(2, interpolation="nearest")(proj_x32)
    fused_x16 = layers.Add()([up_x32, proj_x16])
    
    # Step B: Upsample fused x16 to x8 resolution and sum
    up_x16 = layers.UpSampling2D(2, interpolation="nearest")(fused_x16)
    fused_x8 = layers.Add()([up_x16, proj_x8])
    
    # Step C: Upsample fused x8 to x4 resolution and sum
    up_x8 = layers.UpSampling2D(2, interpolation="nearest")(fused_x8)
    fused_x4 = layers.Add()([up_x8, proj_x4])

    # Final feature representation
    feature_map = layers.Conv2D(num_features, kernel_size=3, strides=1, padding="same", use_bias=False)(fused_x4)
    feature_map = layers.BatchNormalization()(feature_map)
    feature_map = layers.ReLU()(feature_map)

    # Instantiate model with base_model's input and our fused output
    model = Model(inputs=base_model.input, outputs=feature_map, name="MobileNetV3_Small_CenterNet")
    return model
