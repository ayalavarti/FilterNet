import tensorflow as tf
import nn.hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

    @tf.function
    def call(self, inputs):
        pass

    @tf.function
    def loss_function(self, disc_model_output):
        pass


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

    @tf.function
    def call(self, inputs):
        pass

    @tf.function
    def loss_function(self, disc_expert_output, disc_model_output):
        pass
