import tensorflow as tf
from tensorflow.python.keras.layers import LeakyReLU

import nn.hyperparameters as hp
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPool2D, Dropout, \
    Flatten, Dense, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import Adam


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.L = hp.L
        self.K = hp.K
        self.batch_size = hp.batch_size

        init = tf.keras.initializers.RandomNormal

        # Adam optimizer with 1e-4 lr
        self.optimizer = Adam(learning_rate=hp.learning_rate)
        # LeakyReLU activation with alpha=0.2
        self.leaky_relu = LeakyReLU(hp.alpha)

        # ====== Shared convolution layers ======
        self.conv0 = Conv2D(16, (3, 3), strides=(2, 2), padding='same',
                            input_shape=(hp.img_size, hp.img_size, 3))
        self.conv1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')
        self.conv2 = Conv2D(32, (3, 3), strides=(2, 2), padding='same')
        self.conv3 = Conv2D(48, (2, 2), strides=(2, 2), padding='same')
        self.conv4 = Conv2D(48, (2, 2), strides=(2, 2), padding='same')
        self.conv5 = Conv2D(64, (2, 2), strides=(2, 2), padding='same')
        self.conv6 = Conv2D(self.L+12, (2, 2), strides=(2, 2), padding='same')

        # ====== BatchNormalization layers ======
        self.batch_norm_2 = BatchNormalization()
        self.batch_norm_3 = BatchNormalization()
        self.batch_norm_4 = BatchNormalization()
        self.batch_norm_5 = BatchNormalization()
        self.batch_norm_6 = BatchNormalization()

        # ====== Value Network Layers ======
        self.flatten = Flatten()
        self.dense_1 = Dense(1)

        # ====== Policy Network Layers ======
        self.avg_pool = AveragePooling2D((2, 2), strides=(2, 2))
        # Policy convolution layers
        self.conv7 = Conv1D(16, 3)
        self.conv8 = Conv1D(32, 3)
        self.conv9 = Conv1D(48, 3)
        self.conv10 = Conv1D(48, 3)
        self.conv11 = Conv1D(64, 3)
        self.conv12 = Conv1D(self.K, 3)
        # BatchNormalization layers
        self.batch_norm_7 = BatchNormalization()
        self.batch_norm_8 = BatchNormalization()
        self.batch_norm_9 = BatchNormalization()
        self.batch_norm_10 = BatchNormalization()
        self.batch_norm_11 = BatchNormalization()

    @tf.function
    def call(self, state, testing=False):
        h = self.leaky_relu(self.conv0(state))
        h = self.leaky_relu(self.conv1(h))
        h = self.leaky_relu(self.batch_norm_2(self.conv2(h)))
        h = self.leaky_relu(self.batch_norm_3(self.conv3(h)))
        h = self.leaky_relu(self.batch_norm_4(self.conv4(h)))
        h = self.leaky_relu(self.batch_norm_5(self.conv5(h)))
        h = self.leaky_relu(self.batch_norm_6(self.conv6(h)))
        # Return policy (probabilities, actions), value
        return self.get_policy(h, det=testing), self.get_value(h)

    # @tf.function
    def get_policy(self, h, det=False):
        # Global average poling and transpose
        p = self.avg_pool(h)
        p = tf.reshape(p, (self.batch_size, -1, 1))
        p = self.leaky_relu(self.batch_norm_7(self.conv7(p)))
        p = self.leaky_relu(self.batch_norm_8(self.conv8(p)))
        p = self.leaky_relu(self.batch_norm_9(self.conv9(p)))
        p = self.leaky_relu(self.batch_norm_10(self.conv10(p)))
        p = self.leaky_relu(self.batch_norm_11(self.conv11(p)))
        p = self.conv12(p)

        # Get softmax distribution for each a_k (column of p) corresponding to
        # all L steps
        soft_dist = tf.nn.softmax(p, axis=1)

        if det:
            # Return for each image in batch the K parameters by argmax over
            # dimension 1 or the distribution of L steps for the kth filter
            act = tf.argmax(soft_dist, axis=1)
        else:
            # For all K filters, sample from all batches of all L timesteps to
            # determine parameter choice for the kth filter
            act = [tf.random.categorical(soft_dist[:, :, i], 1) for i in range(self.K)]
            act = tf.convert_to_tensor(act)
            act = tf.transpose(tf.squeeze(act))

        soft_dist = tf.transpose(soft_dist, (0, 2, 1))
        return soft_dist, act

    @tf.function
    def get_value(self, image):
        return self.dense_1(self.flatten(image))

    @tf.function
    def loss_function(self, disc_model_output):
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.ones([hp.batch_size, 1]),
                                                disc_model_output))


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Adam optimizer with 1e-4 lr
        # consider switching to RMSprop with no momentum and lr of 0.00005
        self.optimizer = Adam(learning_rate=hp.learning_rate)
        # LeakyReLU activation with alpha=0.2
        self.leaky_relu = LeakyReLU(hp.alpha)
        self.batch_size = hp.batch_size
        self.lda = hp.lda

        # ====== Discriminator layers ======
        self.conv1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')
        self.conv2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.conv3 = Conv2D(48, (2, 2), strides=(2, 2), padding='same')
        self.conv4 = Conv2D(48, (2, 2), strides=(2, 2), padding='same')
        self.conv5 = Conv2D(64, (2, 2), strides=(2, 2), padding='same')
        self.conv6 = Conv2D(48, (2, 2), strides=(2, 2), padding='same')
        self.flatten = Flatten()
        self.dense_1 = Dense(1)

    @tf.function
    def call(self, state):
        v = self.leaky_relu(self.conv1(state))
        v = self.leaky_relu(self.conv2(v))
        v = self.leaky_relu(self.conv3(v))
        v = self.leaky_relu(self.conv4(v))
        v = self.leaky_relu(self.conv5(v))
        v = self.leaky_relu(self.conv6(v))
        v = self.dense_1(self.flatten(v))
        return v

    @tf.function
    def loss_function(self, disc_expert_output, disc_model_output):
        real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones([hp.batch_size, 1]), disc_expert_output))
        fake_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.zeros([hp.batch_size, 1]), disc_model_output))
        return real_loss + fake_loss
