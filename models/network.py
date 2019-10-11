import tensorflow as tf
import tensorflow.keras as K

class baseline_backbone():

    def  __init__(self, backbone_name="resnet50", image_size=(224, 224), training=True):

        input_tensor = K.Input(shape=(image_size[0], image_size[1], 3))

        if backbone_name=="resnet50":
            self.base_model = K.applications.ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
        elif backbone_name=="dense101":
            self.base_model = K.applications.DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False)
        elif backbone_name=="xception":
            self.base_model = K.applications.Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)
        else:
            raise NotImplementedError

        self.global_avg_pool = K.layers.GlobalAveragePooling2D()
        self.dense = K.layers.Dense(3)
        # self.output_act = K.layers.Activation(K.activations.exponential)

    def build_model(self):

        feature = self.base_model.output
        x = self.global_avg_pool(feature)
        x = self.dense(x)
        # score = tf.math.exp(x)

        return K.Model(inputs=self.base_model.input, outputs=x)


class meta_model():

    def __init__(self, backbone_name="resnet50", image_size=(224,224), training=True):

        input_tensor = K.Input(shape=(image_size[0], image_size[1], 3))

        if backbone_name == "resnet50":
            self.base_model = K.applications.ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
        elif backbone_name == "dense101":
            self.base_model = K.applications.DenseNet121(input_tensor=input_tensor, weights='imagenet',
                                                         include_top=False)
        elif backbone_name == "xception":
            self.base_model = K.applications.Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)
        else:
            raise NotImplementedError

        self.conv2d_1 = K.layers.Conv2d(filter=256, kernel_size=(3,3), strides=1, padding="same", activation='relu',
                                        kernel_initializer='he_normal', bias_initializer='zeros')

        self.conv2d_2 = K.layers.Conv2d(filter=256, kernel_size=(3, 3), strides=1, padding="same", activation='relu',
                                        kernel_initializer='he_normal', bias_initializer='zeros')

        self.conv2d_3 = K.layers.Conv2d(filter=256, kernel_size=(3, 3), strides=1, padding="same", activation='relu',
                                        kernel_initializer='he_normal', bias_initializer='zeros')

        self.global_avg_pool = K.layers.GlobalAveragePooling2D()
        self.dense = K.layers.Dense(1, kernel_initializer='he_uniform', activation='sigmoid', use_bias=False)
        # self.output_act = K.layers.Activation(K.activations.exponential)

    def build_model(self, mode):

        assert mode in ["source", "target"], "mode must be either 'source' or 'target'"

        # source task - classification
        feature = self.base_model.output
        x = self.global_avg_pool(feature)
        prob = self.dense(x)

        # target task - PASI
        for layer in self.base_model.layers:
            layer.trainable = False

        if mode == "source":
            return K.model(input=self.base_model.input, output=prob)

        return K.Model(inputs=self.base_model.input, outputs=x)

