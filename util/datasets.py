import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import nn.hyperparameters as hp


def standardize(img, mean, stdev):
	pass


class Datasets:
	"""
	Class for containing the training and test sets.
	"""

	def __init__(self, untouched_data_path, edited_data_path, task):
		self.task = task
		pass

	def calc_mean_and_std(self):
		pass

	def preprocess_fn(self, img):
		pass

	def get_data(self):
		"""
		Returns an image data generator. Batch data generator should return a
		tuple of raw and edited images.
		"""
		pass
