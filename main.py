import os
from argparse import ArgumentParser, ArgumentTypeError

import tensorflow as tf

import nn.hyperparameters as hp
from nn.models import Generator, Discriminator
from util.data.preprocess import Datasets

# Killing optional CPU driver warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
	def valid_dir(directory):
		if os.path.isdir(directory):
			return os.path.normpath(directory)
		else:
			raise ArgumentTypeError(f"Invalid directory: {directory}")

	def valid_model(model_file):
		if model_file.endswith(".h5"):
			return os.path.normpath(model_file)
		else:
			raise ArgumentTypeError(f"Invalid model file: {model_file}")

	parser = ArgumentParser(
		prog="FilterNet",
		description="A deep learning program for photo-realistic image editing")

	subparsers = parser.add_subparsers()
	# Subparser for train command
	tn = subparsers.add_parser(
		"train",
		description="Train a new model on the given untouched and edited images")
	tn.set_defaults(command="train")

	tn.add_argument("--epochs",
					type=int, default=hp.num_epochs,
					help="Number of epochs to train for")

	tn.add_argument("--load-checkpoint",
					default=None,
					help="Path to model checkpoint file (should end with the"
						 "extension .h5)")

	tn.add_argument("--checkpoint-dir",
					default=os.getcwd() + "/model_weights",
					help="Directory to store checkpoint model weights")

	tn.add_argument("--untouched-dir",
					type=valid_dir,
					default=os.getcwd() + "/data/train/untouched",
					help="Directory of untouched images for training")

	tn.add_argument("--edited-dir",
					type=valid_dir,
					default=os.getcwd() + "/data/train/edited/C",
					help="Directory of expert edited images for training")

	# Subparser for test command
	ts = subparsers.add_parser(
		"test", description="Evaluate the model on the given test data")
	ts.set_defaults(command="test")

	ts.add_argument("--checkpoint-file",
					type=valid_model,
					help="Model weights to use with testing")

	ts.add_argument("--untouched-dir",
					type=valid_dir,
					default=os.getcwd() + "/data/test/untouched",
					help="Directory of untouched images for testing")

	ts.add_argument("--edited-dir",
					type=valid_dir,
					default=os.getcwd() + "/data/test/edited",
					help="Directory of expert edited images for testing")

	return parser.parse_args()


# Make arguments global
ARGS = parse_args()


def train():
	pass


def test():
	pass


def main():
	gpu_available = tf.test.is_gpu_available()

	pass


if __name__ == "__main__":
	main()
