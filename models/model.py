import tensorflow as tf

class Generator_Model(tf.keras.Model):
    def __init__(self):
        super(Generator_Model, self).__init__()
        pass

    @tf.function
    def call(self, inputs):
        pass

    @tf.function
    def loss_function(self, disc_model_output):
        pass


class Discriminator_Model(tf.keras.Model):
    def __init__(self):
        super(Discriminator_Model, self).__init__()
        pass

    @tf.function
    def call(self, inputs):
        pass

    def loss_function(self, disc_expert_output, disc_model_output):
        pass
