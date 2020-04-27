import tensorflow as tf
from tensorflow.python.keras.layers import LeakyReLU

import nn.hyperparameters as hp
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv1D, Dense, BatchNormalization, \
	Flatten, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


class Generator(tf.keras.Model):
	def __init__(self):
		super(Generator, self).__init__()
		self.L = hp.L
		self.K = hp.K
		self.batch_size = hp.batch_size

		# Loss hyperparameters
		self.alpha = hp.alpha
		self.beta = hp.beta

		self.a_min = hp.ak_min
		self.a_max = hp.ak_max

		# Adam optimizer with 1e-4 lr
		self.optimizer = Adam(learning_rate=hp.learning_rate, beta_1=hp.beta_1,
							  clipnorm=40)
		# LeakyReLU activation with alpha=0.2
		self.leaky_relu = LeakyReLU(hp.lr_alpha)

		# ====== Shared convolution layers ======
		#self.conv0 = Conv2D(16, (3, 3), strides=(2, 2), padding='same')
		self.conv1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')
		self.conv2 = Conv2D(32, (3, 3), strides=(2, 2), padding='same')
		self.conv3 = Conv2D(48, (2, 2), strides=(2, 2), padding='same')
		self.conv4 = Conv2D(48, (2, 2), strides=(2, 2), padding='same')
		self.conv5 = Conv2D(64, (2, 2), strides=(2, 2), padding='same')
		self.conv6 = Conv2D(self.L + 12, (2, 2), strides=(2, 2), padding='same')

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
		#h = self.leaky_relu(self.conv0(state))
		h = self.leaky_relu(self.conv1(state))
		h = self.leaky_relu(self.batch_norm_2(self.conv2(h)))
		h = self.leaky_relu(self.batch_norm_3(self.conv3(h)))
		h = self.leaky_relu(self.batch_norm_4(self.conv4(h)))
		h = self.leaky_relu(self.batch_norm_5(self.conv5(h)))
		h = self.leaky_relu(self.batch_norm_6(self.conv6(h)))
		# Return policy (probabilities, actions), value
		return self.get_policy(h, det=testing), self.get_value(h)

	@tf.function
	def get_policy(self, h, det=False):
		# Global average poling and transpose
		p = self.avg_pool(h)
		p = tf.reshape(p, (len(h), -1, 1))
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

	def scale_action_space(self, act):
		return self.a_min + (self.a_max - self.a_min) * ((act - 1) / (self.L - 1))

	@tf.function
	def loss_function(self, state, y_model, d_model):
		# Reward
		R = d_model - self.alpha * tf.reduce_mean(tf.square(state - y_model))

		# ========= A2C RL training =========
		(prob, act), value = self.call(state)

		# ======= Value Loss =======
		advantage = R - value
		value_loss = advantage ** 2

		# ======= Policy Loss =======
		# One hot encode actions for all L steps
		action_one_hot = tf.one_hot(act, self.L, dtype=tf.float32)
		# Entropy of filter's prob dist for each img, sum over filters and steps
		entropy = tf.reduce_sum(prob * tf.math.log(prob + 1e-20), axis=[1, 2])
		entropy = tf.reshape(entropy, (-1, 1))

		# Cross-entropy for multi-class exclusive problem sum down all filters
		#policy_loss = -1 * tf.reduce_sum(tf.math.log(prob), axis=[1, 2])
		#policy_loss = tf.reshape(policy_loss, (-1, 1))
		policy_loss = tf.reduce_sum(categorical_crossentropy(
			action_one_hot, prob), axis=1, keepdims=True)

		# Stop gradient flow from value network with advantage calculation
		policy_loss *= tf.stop_gradient(advantage)
		policy_loss -= self.beta * entropy

		total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
		return total_loss


class Discriminator(tf.keras.Model):
	def __init__(self):
		super(Discriminator, self).__init__()
		# Loss hyperparameters
		self.lda = hp.lda

		# Adam optimizer with 1e-4 lr
		self.optimizer = Adam(learning_rate=hp.learning_rate, beta_1=hp.beta_1,
							  clipnorm=40)

		# LeakyReLU activation with alpha=0.2
		self.leaky_relu = LeakyReLU(hp.lr_alpha)
		self.batch_size = hp.batch_size

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

	def _interpolate(self, a, b):
		shape = [self.batch_size, 1, 1, 1]
		eps = tf.random.uniform(shape=shape, minval=0., maxval=1.)
		return a + eps * (b - a)

	@tf.function
	def _gradient_penalty(self, expert, model):
		x = self._interpolate(expert, model)
		grad_interpolated = tf.gradients(self.call(x), [x])[0]
		grad_norm = tf.sqrt(1e-8 + tf.reduce_sum(
			tf.square(grad_interpolated), axis=[1, 2, 3]))
		gp = tf.reduce_mean((grad_norm - 1.) ** 2)
		return gp

	@tf.function
	def loss_function(self, y_model, y_expert, d_model, d_expert):
		# WGAN discriminator loss
		wgan_disc_loss = tf.reduce_mean(d_model) - tf.reduce_mean(d_expert)
		# Gradient penalty
		gp = self._gradient_penalty(y_expert, y_model)
		return wgan_disc_loss + self.lda * gp
