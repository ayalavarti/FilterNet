import os
from argparse import ArgumentParser, ArgumentTypeError

import tensorflow as tf

import nn.hyperparameters as hp
import util.sys as sys
from nn.models import Generator, Discriminator
from util.datasets import Datasets

# Killing optional CPU driver warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpu_available = tf.config.list_physical_devices("GPU")


def parse_args():
	def valid_dir(directory):
		if os.path.isdir(directory):
			return os.path.normpath(directory)
		else:
			raise ArgumentTypeError(f"Invalid directory: {directory}")

	parser = ArgumentParser(
		prog="FilterNet",
		description="A deep learning program for photo-realistic image editing")

	parser.add_argument(
		"--checkpoint-dir",
		default=os.getcwd() + "/model_weights",
		help="Directory to store checkpoint model weights")

	parser.add_argument(
		"--device",
		type=str,
		default="GPU:0" if gpu_available else "CPU:0",
		help="Specify the device of computation eg. CPU:0, GPU:0, GPU:1, ... ")

	subparsers = parser.add_subparsers()
	subparsers.required = True
	subparsers.dest = "command"

	# Subparser for train command
	tn = subparsers.add_parser(
		"train",
		description="Train a new model on the given untouched and edited images")
	tn.set_defaults(command="train")

	tn.add_argument(
		"--epochs",
		type=int, default=hp.num_epochs,
		help="Number of epochs to train for")

	tn.add_argument(
		"--restore-checkpoint",
		action="store_true",
		help="Use this flag to resuming training from a previous checkpoint")

	tn.add_argument(
		"--untouched-dir",
		type=valid_dir,
		default=os.getcwd() + "/data/train/untouched",
		help="Directory of untouched images for training")

	tn.add_argument(
		"--edited-dir",
		type=valid_dir,
		default=os.getcwd() + "/data/train/edited",
		help="Directory of expert edited images for training")

	tn.add_argument(
		"--editor",
		choices=['A', 'B', 'C', 'D', 'E'],
		default='C',
		help="Which editor images to use for training")

	# Subparser for test command
	ts = subparsers.add_parser(
		"test", description="Test the model on the given test data")
	ts.set_defaults(command="test")

	ts.add_argument(
		"--untouched-dir",
		type=valid_dir,
		default=os.getcwd() + "/data/test/untouched",
		help="Directory of untouched images for testing")

	ts.add_argument(
		"--edited-dir",
		type=valid_dir,
		default=os.getcwd() + "/data/test/edited",
		help="Directory of expert edited images for testing")

	ts.add_argument(
		"--editor",
		choices=['C'],
		default='C',
		help="Which editor images to use for testing")

	# Subparser for evaluate command
	ev = subparsers.add_parser(
		"evaluate", description="Evaluate / run the model on the given data")
	ev.set_defaults(command="evaluate")

	ev.add_argument(
		"--image-dir",
		type=valid_dir,
		default=os.getcwd() + "/data/test/untouched",
		help="Directory of untouched images for evaluating")

	ev.add_argument(
		"--output-dir",
		type=valid_dir,
		default=os.getcwd() + "/output/",
		help="Directory of output edited images for testing")

	return parser.parse_args()


# Make arguments global
ARGS = parse_args()


def train():
	pass


def test():
	pass


def main():
	# Initialize generator and discriminator models
	generator = Generator()
	discriminator = Discriminator()

	# Ensure the checkpoint directory exists
	sys.enforce_dir(ARGS.checkpoint_dir)

	# Set up tf checkpoint manager
	checkpoint = tf.train.Checkpoint(
		generator=generator,
		discriminator=discriminator)

	manager = tf.train.CheckpointManager(
		checkpoint, ARGS.checkpoint_dir,
		max_to_keep=3)

	try:
		with tf.device("/device:" + ARGS.device):
			if ARGS.command == "train":
				# train here!
				dataset = Datasets(
					ARGS.untouched_dir, ARGS.edited_dir, "train", ARGS.editor)
				pass

			if ARGS.command == 'test':
				# test here!
				dataset = Datasets(
					ARGS.untouched_dir, ARGS.edited_dir, "test", ARGS.editor)
				pass

			if ARGS.command == 'evaluate':
				# Ensure the output directory exists
				sys.enforce_dir(ARGS.output_dir)
				# evaluate here!
				dataset = Datasets(ARGS.image_dir, None, "evaluate")
				pass

	except RuntimeError as e:
		# something went wrong should not get here
		print(e)


if __name__ == "__main__":
	main()
