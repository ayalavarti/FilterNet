import os

from PIL import Image

import nn.hyperparameters as hp
import numpy as np

from skimage.transform import resize
from util.lightroom.editor import PhotoEditor
from resizeimage import resizeimage


def enforce_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def edit_original(big_image, generator):
    """
    Takes in a full-sized image and runs the generator on a smaller version.
    Returns an edited version of the full-sized image.
    """
    resized = resize(big_image, (hp.img_size, hp.img_size)).astype(np.float32)

    prob, _ = generator(resized[None])
    act_scaled, _ = generator.convert_prob_act(prob.numpy(), det=True,
                                               det_avg=hp.det_avg)
    print(big_image.shape)
    if big_image.shape[0] > 600:
        resized = resizeimage.resize_height(Image.fromarray(big_image), 600)
    elif big_image.shape[1] > 600:
        resized = resizeimage.resize_width(Image.fromarray(big_image), 600)
    resized = np.array(resized)
    orig_edit = PhotoEditor.edit((resized/255)[None], act_scaled)
    return orig_edit[0]
