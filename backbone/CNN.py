from tensorflow.keras.applications import  VGG19,EfficientNetB0,InceptionV3,ResNet50,EfficientNetB3
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Lambda, Conv2D, \
      BatchNormalization, MaxPool2D, Reshape, Dense, Softmax,\
      Rescaling, Dropout, AveragePooling2D, Flatten, MaxPooling2D
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, input_shape: list, hiden: int, ps: list, ss: list, prob_drop = 0.5,backbone = 'VGG19',**kwargs):
        #ps is pool size
        #ss is stride size

        super().__init__()
        h,w = input_shape
        input_shape_ = [h,w,3]
        if backbone == 'VGG19':
            self.features_extraction = VGG19(input_shape = input_shape_, include_top = False, weights = 'imagenet')
        elif backbone == "ResNet50":
            self.features_extraction = ResNet50(input_shape = input_shape_, include_top = False, weights = 'imagenet')
        elif backbone == "InceptionV3":
            self.features_extraction = InceptionV3(input_shape = input_shape_, include_top = False, weights = 'imagenet')

        self.Lambda = Lambda(lambda x : tf.stack([x] * 3, axis=-1))
        # def transpose(x):
        #     return tf.transpose(x, )
        # self.transpose =
        self.dropout = Dropout(prob_drop)
        self.pointwise = Conv2D(filters = hiden, kernel_size = 1)

        def reshape(a):
            return tf.reshape(tf.transpose(a, perm=[0, 3, 2, 1]), shape=(-1, a.shape[3], a.shape[1]*a.shape[2]))
        self.reshape = reshape

    def call(self, inputs):
        inputs = self.Lambda(inputs)

        conv = self.features_extraction(inputs)
        #b h w c
        conv = self.dropout(conv)
        #b h w c
        conv = self.pointwise(conv)
        #b h w c'(= number of hidden channels)
        conv = self.reshape(conv)
        #b c' (hxw)
        return conv




def CNN(backbone, input_shape, **kwargs):
    if backbone == 'VGG19':
        model = VGG19(input_shape = input_shape, include_top = False, weights = 'imagenet')
    elif backbone == "ResNet50":
        model = ResNet50(input_shape = input_shape, include_top = False, weights = 'imagenet')
    elif backbone == "InceptionV3":
        model = InceptionV3(input_shape = input_shape, include_top = False, weights = 'imagenet')
    return model

# def Transformer():

def Predictor(input_imgs, backbone='VGG19', transformer= None, dropout = 0.1):
    x = Rescaling(1.0 / 255)(input_imgs)
    x = CNN(backbone=backbone, input_shape=(800, 100, 3))(x)
    x = Dropout(dropout)(x)
    x = AveragePooling2D(pool_size = (1,3))(x)
    x = Conv2D(filters = 1, kernel_size = (1,1))(x)
    outputs = Flatten()(x)
    return outputs



