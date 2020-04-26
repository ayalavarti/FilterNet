import os
from util.datasets import Datasets
from skimage import io
from util.lightroom.editor import PhotoEditor
import numpy as np
import matplotlib.pyplot as plt

UNTOUCHED_TRAIN = './sample_data/train/untouched'
EDITED_TRAIN = './sample_data/train/edited'

UNTOUCHED_TEST = './sample_data/test/untouched'
EDITED_TEST = './sample_data/test/edited'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

td = Datasets(UNTOUCHED_TRAIN, EDITED_TRAIN, 'train')

parameters = np.array(
	[[.125, .375, -.223, .125, .125, .125, .125, .125, .125, .125, .125],
	 [.125, .375, .123, .125, .125, .125, .125, .125, .125, .125, .125]])


def lightroom_one_filter():
	"""
    Given a filter applies the filter, and displays the edited and unedited
    versions of first image of the first batch
    """
	for b in td.data:
		raw_photos = b.numpy()[:2, 0]
		edited_photos = PhotoEditor.edit(raw_photos, parameters, ind=[2])
		io.imshow(raw_photos[0])
		io.show()
		io.imshow(edited_photos[0])
		io.show()
		plt.show()
		break


def lightroom_all_filters():
	for b in td.data:
		raw_photos = b.numpy()[:2, 0]
		edited_photos = PhotoEditor.edit(raw_photos, parameters)
		io.imshow(raw_photos[0])
		io.show()
		io.imshow(edited_photos[0])
		io.show()
		plt.show()
		break


def test_lightroom():
	"""
    Applies all filters outlined in photo editor to training set
    """
	for b in td.data:
		raw_photos = b.numpy()[:2, 0]
		edited_photos = PhotoEditor.edit(raw_photos, parameters)
		assert (not np.array_equal(raw_photos, edited_photos))
		for i in range(0, 4):
			assert (raw_photos.shape[i] == edited_photos.shape[i])


if __name__ == '__main__':
	lightroom_one_filter()
	# test_lightroom()
