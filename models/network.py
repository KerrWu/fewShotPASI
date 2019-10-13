import tensorflow as tf
import tensorflow.keras as K


class baseline_backbone():
    def __init__(self, backbone_name="resnet50", image_size=(224, 224), training=True):

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

        self.global_avg_pool = K.layers.GlobalAveragePooling2D()
        self.dense = K.layers.Dense(3, name="three_scores", kernel_initializer='he_uniform', use_bias=False)
        # self.output_act = K.layers.Activation(K.activations.exponential)

    def build_model(self):

        feature = self.base_model.output
        x = self.global_avg_pool(feature)
        x = self.dense(x)
        # score = tf.math.exp(x)

        return K.Model(inputs=self.base_model.input, outputs=x)


class meta_model():
    def __init__(self, image_size=(224, 224), backbone_name="resnet50", training=True):

        input_tensor = K.layers.Input(shape=(image_size[0], image_size[1], 3))
        if backbone_name == "resnet50":
            self.base_model = K.applications.ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
        # elif backbone_name == "dense101":
        #     self.base_model = K.applications.DenseNet121(weights='imagenet',
        #                                                  include_top=False)
        # elif backbone_name == "xception":
        #     self.base_model = K.applications.Xception(weights='imagenet', include_top=False)
        else:
            raise NotImplementedError

        self.global_avg_pool = K.layers.GlobalAveragePooling2D()
        self.dense_prob = K.layers.Dense(1, kernel_initializer='he_uniform', activation='sigmoid', use_bias=False,
                                         name="output_prob")
        self.dense_score = K.layers.Dense(3, kernel_initializer='he_uniform', use_bias=False, name="output_scores")
        # self.output_act = K.layers.Activation(K.activations.exponential)

        self.conv2d_1 = K.layers.Conv2d(filter=256, kernel_size=(3, 3), strides=1, padding="same", activation='relu',
                                        kernel_initializer='he_normal', bias_initializer='zeros')

        self.conv2d_2 = K.layers.Conv2d(filter=256, kernel_size=(3, 3), strides=1, padding="same", activation='relu',
                                        kernel_initializer='he_normal', bias_initializer='zeros')

        self.conv2d_3 = K.layers.Conv2d(filter=256, kernel_size=(3, 3), strides=1, padding="same", activation='relu',
                                        kernel_initializer='he_normal', bias_initializer='zeros')

    def build_model(self, mode, meta_weights=None):

        assert mode in ["source", "target"], "mode must be either 'source' or 'target'"

        # source task - classification
        if mode == "source":
            feature = self.base_model.output
            embedding = self.global_avg_pool(feature)
            prob = self.dense_prob(embedding)

            return K.model(input=self.base_model.input, output=[embedding, prob])

        # target task - PASI
        if mode == 'target':

            assert meta_weights != None, "meta weights must be provided for target task"

            feature = self.base_model.output
            embedding = self.global_avg_pool(feature)
            prob = self.dense_prob(embedding)

            meta_model = K.model(input=self.base_model.input, output=[feature, prob])
            meta_model.load_weights(meta_weights)

            for layer in meta_model.layers:
                layer.trainable = False

            meta_feature = meta_model.output[0]

            x = self.conv2d_1(meta_feature)
            x = self.conv2d_2(x)
            x = self.conv2d_3(x)

            x = self.global_avg_pool(x)
            scores = self.dense_score(x)

            return K.Model(inputs=self.base_model.input, outputs=scores)
