import os
from util.datasets import Datasets
from nn.models import Generator, Discriminator
import nn.hyperparameters as hp


UNTOUCHED_TRAIN = './sample_data/train/untouched'
EDITED_TRAIN = './sample_data/train/edited'

UNTOUCHED_TEST = './sample_data/test/untouched'
EDITED_TEST = './sample_data/test/edited'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

td = Datasets(UNTOUCHED_TRAIN, EDITED_TRAIN, 'train')


def test_discriminator_model():
	discriminator = Discriminator()
	for b in td.data:
		disc_output = discriminator(b[:, 0])
		assert(disc_output.shape == (hp.batch_size, 1))
		break


def test_discriminator_loss():
	discriminator = Discriminator()
	for b in td.data:
		x_model, x_expert = b[:, 0], b[:, 1]
		d_model, d_expert = discriminator(x_model), discriminator(x_expert)

		loss = discriminator.loss_function(x_model, x_expert, d_model, d_expert)
		print(loss)
		break


def test_generator_loss():
	generator = Generator()
	discriminator = Discriminator()
	for b in td.data:
		x_model, x_expert = b[:, 0], b[:, 1]

		d_model = discriminator(x_model)
		loss = generator.loss_function(x_model, x_model, d_model)
		print(loss)
		break


def test_generator_model():
	generator = Generator()

	for b in td.data:
		policy, value = generator(b[:, 0])
		prob, act = policy
		assert(prob.shape == (hp.batch_size, hp.K, hp.L))
		assert(act.shape == (hp.batch_size, hp.K))
		assert(value.shape == (hp.batch_size, 1))
		break


if __name__ == '__main__':
	test_generator_loss()
