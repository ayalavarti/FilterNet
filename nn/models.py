import tensorflow as tf
from tensorflow.python.keras.layers import LeakyReLU

import nn.hyperparameters as hp
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization,\
    AveragePooling2D


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.L = 0 # num actions we are taking
        self.action_size = 0 # shown as K in graphic in paper, num filters possible
        self.stddev = 0.02
        self.alpha = 0.2

        # Global Layers
        self.leaky_relu = LeakyReLU(self.alpha) # seems they just used relu here
        self.conv1 = Conv2D(16, (3,3), strides=(1, 1), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.conv2 = Conv2D(32, (3,3), strides=(2, 2), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_1 = BatchNormalization(32)
        self.conv3 = Conv2D(48, (2, 2), strides=(2, 2), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_2 = BatchNormalization(48)
        self.conv4 = Conv2D(48, (2, 2), strides=(2, 2), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_3 = BatchNormalization(48)
        self.conv5 = Conv2D(64, (2, 2), strides=(2, 2), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_4 = BatchNormalization(64)
        self.conv6 = Conv2D(self.L + 12, (2, 2), strides=(2, 2), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_5 = BatchNormalization(self.L + 12)

        # Value Network Layers
        self.flatten = Flatten()
        self.dense_1 = Dense(1)
        # Policy Network Layers
        self.avg_pool = AveragePooling2D()
        self.conv7 = Conv1D(16, (3), strides=(1), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_6 = BatchNormalization(16)
        self.conv8 = Conv1D(32, (3), strides=(1), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_7 = BatchNormalization(32)
        self.conv9 = Conv1D(48, (3), strides=(1), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_8 = BatchNormalization(48)
        self.conv10 = Conv1D(48, (3), strides=(1), padding='same',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_9 = BatchNormalization(48)
        self.conv11 = Conv1D(64, (3), strides=(1), padding='same',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev))
        self.batch_norm_10 = BatchNormalization(64)
        self.conv12 = Conv1D(self.action_size, (3), strides=(1), padding='same',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev),
                             activation='softmax')

    @tf.function
    def call(self, inputs):
        output = self.leaky_relu(self.conv1(inputs))
        output = self.leaky_relu(self.batch_norm_1(self.conv2(output)))
        output = self.leaky_relu(self.batch_norm_2(self.conv3(output)))
        output = self.leaky_relu(self.batch_norm_3(self.conv4(output)))
        output = self.leaky_relu(self.batch_norm_4(self.conv5(output)))
        output = self.leaky_relu(self.batch_norm_5(self.conv6(output)))
        return self.get_policy(output), self.get_value(output)

    @tf.function
    def get_policy(self, input):
        output = self.avg_pool(input)
        output = output.reshape((-1, 1))
        output = self.leaky_relu(self.batch_norm_6(self.conv7(output)))
        output = self.leaky_relu(self.batch_norm_7(self.conv8(output)))
        output = self.leaky_relu(self.batch_norm_8(self.conv9(output)))
        output = self.leaky_relu(self.batch_norm_9(self.conv10(output)))
        output = self.leaky_relu(self.batch_norm_10(self.conv11(output)))
        output = self.conv12(output)
        return output

    @tf.function
    def get_value(self, input):
        return self.dense_1(self.flatten(input))

    @tf.function
    def loss_function(self, disc_model_output):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones([hp.batch_size, 1]), disc_model_output))


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
    def loss_function(self, disc_expert_output, disc_model_output):
        real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones([hp.batch_size, 1]), disc_expert_output))
        fake_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.zeros([hp.batch_size, 1]), disc_model_output))
        return real_loss + fake_loss
