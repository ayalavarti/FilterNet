import os
from util.datasets import Datasets
from skimage import io
import nn.hyperparameters as hp
from util.lightroom.editor import PhotoEditor
from util.lightroom import editor as E
import numpy as np
import matplotlib.pyplot as plt # matplotlib provides plot functions similar to MATLAB


UNTOUCHED_TRAIN = './sample_data/train/untouched'
EDITED_TRAIN = './sample_data/train/edited'

UNTOUCHED_TEST = './sample_data/test/untouched'
EDITED_TEST = './sample_data/test/edited'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

"""given a filter applies the filter, and displays the edited and unedited versions of
 first image of the first batch"""
def plot_lightroom_one_filter(filter):
    td = Datasets(UNTOUCHED_TRAIN, EDITED_TRAIN, 'train')
    for b in td.data:
        raw_photos = b.numpy()[:, 0, :, :, :]
        editor = PhotoEditor()
        editted_photos = editor(raw_photos, {filter: hp.parameters[filter]})
        io.imshow(editted_photos[0, :, :, :])
        io.show()
        io.imshow(raw_photos[0, :, :, :])
        io.show()
        plt.show()
        break;
    assert(1==1)

"""applies all filters outlined in photoeditor to training set"""
def test_lightroom():
    td = Datasets(UNTOUCHED_TRAIN, EDITED_TRAIN, 'train')
    for b in td.data:
        raw_photos = b.numpy()[:,1,:,:,:]
        editor = PhotoEditor()
        editted_photos = editor(raw_photos, hp.parameters)
        assert (np.array_equal(raw_photos, editted_photos) == False)
        for i in range(0, 4):
            assert(raw_photos.shape[i] == editted_photos.shape[i])


if __name__ == '__main__':
    plot_lightroom_one_filter(E.saturation)
    #test_lightroom()
