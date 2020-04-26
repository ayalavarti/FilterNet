import os
import datetime
from argparse import ArgumentParser, ArgumentTypeError

import tensorflow as tf

import nn.hyperparameters as hp
import util.sys as sys
import util.visualize as viz
from nn.models import Generator, Discriminator
from util.datasets import Datasets
from util.lightroom.editor import PhotoEditor

# Killing optional CPU driver warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.backend.set_floatx('float32')

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

	tn.add_argument(
		"--print-every-x-batches",
		type=int, default=hp.print_every_x_batches,
		help="After how many batches you want to print")

	tn.add_argument(
		"--save-every-x-batches",
		type=int, default=hp.save_every_x_batches,
		help="After how many batches you want to save")

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

	ts.add_argument(
		"--display",
		type=bool,
		default=False,
		help="True if you want to display the test output, false to save to file.")

	ts.add_argument(
		"--num-display",
		type=int,
		default=hp.test_images,
		help="Number of test images to display, must be <= batch size")

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


def train(dataset, manager, generator, discriminator):
	log_dir = os.path.join("logs/scalars/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	summary_writer = tf.summary.create_file_writer(logdir=log_dir)

	for e in range(ARGS.epochs):
		print(f"========== Epoch {e} ==========")
		for b, batch in enumerate(dataset.data):
			# Update Generator
			for _ in range(hp.gen_update_freq):
				with tf.GradientTape() as gen_tape:
					x_model = batch[:, 0]
					(prob, act), value = generator(x_model)
					act = generator.scale_action_space(act)

					y_model = PhotoEditor.edit(x_model.numpy(), act.numpy())
					y_model = tf.convert_to_tensor(y_model, dtype=tf.float32)
					d_model = discriminator(x_model)

					gen_loss = generator.loss_function(x_model, y_model, d_model)

				gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
				generator.optimizer.apply_gradients(
					zip(gen_grad, generator.trainable_variables))

			# Update Discriminator
			for _ in range(hp.disc_update_freq):
				with tf.GradientTape() as disc_tape:
					x_model, y_expert = batch[:, 0], batch[:, 1]
					(prob, act), value = generator(x_model)
					act = generator.scale_action_space(act)

					y_model = PhotoEditor.edit(x_model.numpy(), act.numpy())
					y_model = tf.convert_to_tensor(y_model, dtype=tf.float32)
					d_expert = discriminator(y_expert)
					d_model = discriminator(y_model)

					disc_loss = discriminator.loss_function(
						y_model, y_expert, d_model, d_expert)

				disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
				discriminator.optimizer.apply_gradients(
					zip(disc_grad, discriminator.trainable_variables))

			if b % ARGS.print_every_x_batches == 0:
				print(f"Batch: {b} Generator Loss: {gen_loss} Discriminator Loss: {disc_loss}")
				with summary_writer.as_default():
					tf.summary.scalar('disc_loss', disc_loss,
									  step=discriminator.optimizer.iterations)
					tf.summary.scalar('gen_loss', gen_loss,
									  step=generator.optimizer.iterations)

			if b % ARGS.save_every_x_batches == 0:
				manager.save()



def test(dataset, generator):
	for batch in dataset.data:
		x_model = batch[:, 0]
		policy, value = generator(batch)
		prob, act = policy
		y_model = PhotoEditor.edit(x_model, act)
		# Call visualizer to visualize images
		viz.visualize_batch(batch, y_model, ARGS.display, ARGS.num_display)
		break


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

	if ARGS.command != 'train' or ARGS.restore_checkpoint:
		# Restores the latest checkpoint using from the manager
		checkpoint.restore(manager.latest_checkpoint)

	try:
		with tf.device("/device:" + ARGS.device):
			if ARGS.command == "train":
				# train here!
				dataset = Datasets(
					ARGS.untouched_dir, ARGS.edited_dir, "train", ARGS.editor)

				train(dataset, manager, generator, discriminator)

			if ARGS.command == 'test':
				# test here!
				dataset = Datasets(
					ARGS.untouched_dir, ARGS.edited_dir, "test", ARGS.editor)
				test(dataset, generator)

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
