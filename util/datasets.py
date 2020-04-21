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
		self.file_list = list(self._image_file_list(self.u_dir, self.e_dir))

		# Mean and std for standardization
		self.mean = np.zeros(3)
		self.std = np.ones(3)
		self._calc_mean_and_std(self.u_dir, "u")

		self._create_img_lists(editor)
		self.new_img_shape = tf.cast((hp.img_size, hp.img_size), tf.int32)


		self.data = self._get_data() if task == 'evaluate' else self._get_dual_data()

	@staticmethod
	def _image_file_list(untouched_dir, edited_dir):
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
					  "with overlapping untouched and edited images")
			return untouched_images.intersection(edited_images)
		return untouched_images

	def _dual_input_parser(self, u_path, e_path):
		u_img = tf.io.read_file(u_path)
		u_img = tf.io.decode_jpeg(u_img, channels=3)
		u_img = tf.image.convert_image_dtype(u_img, tf.float32)
		u_img_new = tf.image.resize(u_img, self.new_img_shape)

		e_img = tf.io.read_file(e_path)
		e_img = tf.io.decode_jpeg(e_img, channels=3)
		e_img = tf.image.convert_image_dtype(e_img, tf.float32)
		e_img_new = tf.image.resize(e_img, self.new_img_shape)

		return tf.convert_to_tensor([u_img_new, e_img_new])

	def _input_parser(self, u_path):
		u_img = tf.io.read_file(u_path)
		u_img = tf.io.decode_jpeg(u_img, channels=3)
		u_img = tf.image.convert_image_dtype(u_img, tf.float32)
		u_img_new = tf.image.resize(u_img, self.new_img_shape)

		return tf.convert_to_tensor([u_img_new])

	def _create_img_lists(self, h):
		u_imgs = map(lambda x: join(self.u_dir, f"u-{x}"), self.file_list)
		self.u_imgs = tf.constant(list(u_imgs))

		if self.e_dir:
			e_imgs = map(lambda x: join(self.e_dir, f"{h}-{x}"), self.file_list)
			self.e_imgs = tf.constant(list(e_imgs))

	def _calc_mean_and_std(self, file_path, header):
		"""
		Calculate mean and standard deviation of a sample of the dataset
		for standardization.
        """
		file_list = cp(self.file_list)
		random.shuffle(file_list)

		file_list = file_list[:hp.preprocess_sample_size]
		data_sample = np.zeros(
			(hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

		for i, f in enumerate(file_list):
			img = Image.open(f"{getcwd()}/{file_path}/{header}-{f}")
			img = img.resize((hp.img_size, hp.img_size))
			img = np.array(img, dtype=np.float32)
			img /= 255.

			# Grayscale -> RGB
			if len(img.shape) == 2:
				img = np.stack([img, img, img], axis=-1)

			data_sample[i] = img

		self.mean = np.mean(data_sample, axis=(0, 1, 2))
		self.std = np.std(data_sample, axis=(0, 1, 2))

	def _standardize(self, img):
		"""
		Function for applying standardization to an input image.

		:param img: numpy array of shape (image size, image size, 3)
		:return: img - numpy array of shape (image size, image size, 3)
		"""
		return (img - self.mean) / self.std

	def _get_dual_data(self):
		data = tf.data.Dataset.from_tensor_slices((self.u_imgs, self.e_imgs))
		data = data.map(self._dual_input_parser)

		return data

	def _get_data(self):
		data = tf.data.Dataset.from_tensor_slices(self.u_imgs)
		data = data.map(self._input_parser)

		return data
