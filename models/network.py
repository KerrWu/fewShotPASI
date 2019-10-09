import tensorflow.keras as K

class baseline_backbone(K.Model):

    def  __init__(self):
        super(baseline_backbone, self).__init__()
        self.global_avg_pool = K.layers.GlobalAveragePooling2D()
        self.dense = K.layers.Dense(3)
        self.exp = K.activations.exponential()

    def call(self, inputs, name="resnet50", training=False):

        if name=="resnet50":
            feature = K.applications.ResNet50(input_tensoe=inputs, weights='imagenet', include_top=False).output
        elif name=="dense101":
            feature = K.applications.DenseNet121(input_tensoe=inputs, weights='imagenet', include_top=False).output
        elif name=="xception":
            feature = K.applications.Xception(input_tensoe=inputs, weights='imagenet', include_top=False).output
        else:
            raise NotImplementedError

        x = self.global_avg_pool(feature)
        x = self.dense(x)
        score = self.exp(x)

        return score
