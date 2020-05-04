import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import nn.hyperparameters as hp
from os import listdir, getcwd
from os.path import isfile, join
from copy import deepcopy as cp


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Datasets:
	"""
	Class for containing the training and test sets as well as
	other useful data-related information. Contains the functions
	for pre-processing.
	"""

	def __init__(self, untouched_dir, edited_dir, task, editor='c'):
		self.task = task
		self.u_dir = untouched_dir
		self.e_dir = join(edited_dir, editor.capitalize()) if edited_dir else None

		# Set up file lists of raw images
		self.file_list = self._image_file_list(self.u_dir, self.e_dir)
		self.data_size = len(self.file_list)

		# Set up file path lists of both untouched and edited images
		self._create_img_lists(h=editor)
		self.new_img_shape = tf.cast((hp.img_size, hp.img_size), tf.int32)

		# Set the dataset data attribute using the appropriate get_data func
		self.data = self._get_data() if task == 'evaluate' else self._get_dual_data()

	@staticmethod
	def _image_file_list(untouched_dir, edited_dir):
		"""
		Returns the list of overlapping images from the provided lists of
		image files.

		:param untouched_dir: list of untouched image file paths
		:param edited_dir: list of edited image file paths
		:return: list of overlapping images
		"""
		def valid_img_dir(x, f):
			return isfile(join(x, f)) and f.endswith(".jpg")

		def find_images(x):
			files = [f.split('-')[1] for f in listdir(x) if valid_img_dir(x, f)]
			return set(files)

		untouched_images = find_images(untouched_dir)

		if edited_dir:
			edited_images = find_images(edited_dir)

			if untouched_images - edited_images != set():
				print("Non overlapping images detected. Only using image pairs"
					  " with overlapping untouched and edited images")
			return list(untouched_images.intersection(edited_images))
		return list(untouched_images)

	def _dual_input_parser(self, u_path, e_path):
		"""
		Input parser for tf.data.Dataset containing both untouched and edited
		images.

		:param u_path: filepath to single untouched image
		:param e_path: filepath to single edited image
		:return: tf.tensor of shape [2, hp.img_size, hp.img_size, 3] of both
				untouched, edited images
		"""
		random_flip = tf.random.uniform([1]) > 0.5

		u_img = self._input_parser(u_path)
		e_img = self._input_parser(e_path)

		c_tensor = tf.concat([u_img, e_img], 0)

		return tf.image.flip_left_right(c_tensor) if random_flip else c_tensor

	def _input_parser(self, u_path):
		"""
		Input parser for tf.data.Dataset containing only untouched images.

		:param u_path: filepath to single untouched image
		:return: tf.tensor of shape [1, hp.img_size, hp.img_size, 3] of
				untouched image
		"""
		u_img = tf.io.read_file(u_path)
		u_img = tf.io.decode_jpeg(u_img, channels=3)
		u_img = tf.image.convert_image_dtype(u_img, tf.float32)
		u_img_new = tf.image.resize(u_img, self.new_img_shape)

		return tf.convert_to_tensor([u_img_new])

	def _create_img_lists(self, h=None):
		"""
		Creates lists of untouched and (optionally) edited image file paths
		:param h: header for when processing edited image directory
		"""
		u_imgs = map(lambda x: join(self.u_dir, "u-{}".format(x)), self.file_list)
		self.u_imgs = tf.constant(list(u_imgs))

		if self.e_dir:
			if self.task == "train":
				np.random.shuffle(self.file_list)
			e_imgs = map(lambda x: join(self.e_dir, "{}-{}".format(h.lower(), x)), self.file_list)
			self.e_imgs = tf.constant(list(e_imgs))

	def _get_dual_data(self):
		"""
		Returns a tf.data.Dataset containing 5D tf.Tensors of shape
			[hp.batch_size, 2, hp.img_size, hp.img_size, 3]
		"""
		# Set up tf.data.Dataset with image filepaths of untouched, edited images
		# then map with self._dual_input_parser to convert filepaths to tf.Tensor
		data = tf.data.Dataset.from_tensor_slices((self.u_imgs, self.e_imgs))
		data = data.map(self._dual_input_parser)
		data = data.batch(hp.batch_size)

		return data

	def _get_data(self):
		"""
		Returns a tf.data.Dataset containing 5D tf.Tensors of shape
			[hp.batch_size, 1f, hp.img_size, hp.img_size, 3]
		"""
		# Set up tf.data.Dataset with image filepaths of untouched images
		# then map with self._input_parser to convert filepath to tf.Tensor
		data = tf.data.Dataset.from_tensor_slices(self.u_imgs)
		data = data.map(self._input_parser)
		data = data.batch(hp.batch_size)

		return data
