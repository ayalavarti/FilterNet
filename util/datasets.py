import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import nn.hyperparameters as hp



class Datasets():
	""" Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.

    task: used if we want to do different preprocesses, with different applications
    """

	def __init__(self, untouched_dir, edited_dir, train):

		if train is "train":
			self.train = True
		else:
			self.train = False

		self.untouched_dir = untouched_dir
		self.edited_dir = edited_dir

		# Mean and std for standardization
		self.mean = np.zeros((3,))
		self.std = np.ones((3,))
		self.calc_mean_and_std()

		# Setup data generators
		if self.train:
			self.data = self.get_data(self.data_path, True, True)
		if not self.train:
			self.data = self.get_data(self.data_path, False, False)

	def calc_mean_and_std(self):
		""" Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

		# Get list of all images in training directory
		file_list = []
		for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
			for name in files:
				if name.endswith(".jpg"):
					file_list.append(os.path.join(root, name))

		# Shuffle filepaths
		random.shuffle(file_list)

		# Take sample of file paths
		file_list = file_list[:hp.preprocess_sample_size]

		# Allocate space in memory for images
		data_sample = np.zeros(
			(hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

		# Import images
		for i, file_path in enumerate(file_list):
			img = Image.open(file_path)
			img = img.resize((hp.img_size, hp.img_size))
			img = np.array(img, dtype=np.float32)
			img /= 255.

			# Grayscale -> RGB
			if len(img.shape) == 2:
				img = np.stack([img, img, img], axis=-1)

			data_sample[i] = img

		for i in range(0, 3):
			self.mean[i] = np.mean(data_sample[:, :, :, i])
			self.std[i] = np.std(data_sample[:, :, :, i])

	def standardize(self, img):
		""" Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """
		return (img - self.mean) / self.std

	def preprocess_fn(self, img):
		""" Preprocess function. """
		img = img / 255.
		img = self.standardize(img)
		return img

	def get_data(self, path, shuffle, augment):
		""" Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG preprocessing
                     should be applied to the images.
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

		if augment:
			data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
				rotation_range=5,
				width_shift_range=0.1,
				height_shift_range=0.1,
				horizontal_flip=True,
				preprocessing_function=self.preprocess_fn)
		else:
			data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
				preprocessing_function=self.preprocess_fn)

		# HAVE TO FIX THIS TO ACCOUNT FOR UNTOUCHED AND TOUCHED
		data_gen = data_gen.flow_from_directory(
			path,
			target_size=(hp.img_size, hp.img_size),
			class_mode=None,
			batch_size=hp.batch_size,
			shuffle=shuffle)

		return data_gen
