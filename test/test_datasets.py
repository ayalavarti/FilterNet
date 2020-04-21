import os
from util.datasets import Datasets
from skimage import io


UNTOUCHED_TRAIN = './data/train/untouched'
EDITED_TRAIN = './data/train/edited'


UNTOUCHED_TEST = './data/test/untouched'
EDITED_TEST = './data/test/edited'


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def show_sample():
	td = Datasets(UNTOUCHED_TRAIN, EDITED_TRAIN, 'train')

	for f in td.data.take(1):
		io.imshow(f[0].numpy())
		io.show()

		io.imshow(f[1].numpy())
		io.show()


def test_train_dataset():
	td = Datasets(UNTOUCHED_TRAIN, EDITED_TRAIN, 'train')

	for f in td.data.take(2):
		assert (f.shape[0] == 2)
		assert (f.shape[-1] == 3)


def test_alt_editor_dataset():
	td = Datasets(UNTOUCHED_TRAIN, EDITED_TRAIN, 'train', editor='a')

	for f in td.data.take(2):
		assert (f.shape[0] == 2)
		assert (f.shape[-1] == 3)


def test_test_dataset():
	td = Datasets(UNTOUCHED_TEST, EDITED_TEST, 'test')

	for f in td.data.take(2):
		assert (f.shape[0] == 2)
		assert (f.shape[-1] == 3)


def test_evaluate_dataset():
	td = Datasets(UNTOUCHED_TEST, None, 'evaluate')

	for f in td.data.take(2):
		assert (f.shape[0] == 1)
		assert (f.shape[-1] == 3)


if __name__ == '__main__':
	show_sample()
