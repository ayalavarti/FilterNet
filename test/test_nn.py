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
		disc_output = discriminator(b[None, 0, 0])
		# prob, act = policy
		# assert(prob.shape == (hp.K, hp.L))
		# assert(act.shape == (hp.K, ))
		# assert(value.shape == (1, 1))
		break



def test_generator_model():
	generator = Generator()

	for b in td.data:
		policy, value = generator(b[None, 0, 0])
		prob, act = policy
		assert(prob.shape == (hp.K, hp.L))
		assert(act.shape == (hp.K, ))
		assert(value.shape == (1, 1))
		break


if __name__ == '__main__':
	test_generator_model()
