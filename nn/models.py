import tensorflow as tf
from tensorflow.python.keras.layers import LeakyReLU

import nn.hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

    @tf.function
    def call(self, inputs):
        pass

    @tf.function
    def loss_function(self):
        pass


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(hp.learning_rate)
        alpha = 0.2
        stddev = 0.02
        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(16, (3,3), strides=(1, 1), padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev)))
        self.model.add(LeakyReLU(alpha))
        self.model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev)))
        self.model.add(LeakyReLU(alpha))
        self.model.add(Conv2D(48, (2, 2), strides=(2, 2), padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev)))
        self.model.add(LeakyReLU(alpha))
        self.model.add(Conv2D(48, (2, 2), strides=(2, 2), padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev)))
        self.model.add(LeakyReLU(alpha))
        self.model.add(Conv2D(64, (2, 2), strides=(2, 2), padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev)))
        self.model.add(LeakyReLU(alpha))
        self.model.add(Conv2D(48, (2, 2), strides=(2, 2), padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev)))
        self.model.add(LeakyReLU(alpha))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        # Do we need an activation here?

    @tf.function
    def call(self, inputs):
        return self.model(inputs)

    @tf.function
    def loss_function(self):
        pass
