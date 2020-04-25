import cv2
import numpy as np
from skimage import io, img_as_ubyte, img_as_float32
import math
import sys
from skimage.restoration import denoise_bilateral
from skimage.color import rgb2hsv, hsv2rgb


# from matplotlib.colors import hsv_to_rgb  # skimage didn't have one for some reason
# pip install opencv-python


def relu(vector):
    vector[vector < 0] = 0
    return vector


def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))\


def sigmoid_inversed(vector):  # basically the logit function
    # paper implements this differently than I do someone check to see if i need to check for overflow
    denom = 1-vector
    denom[denom < .001] = .001
    divided = vector / (denom)
    divided[divided<=0]=0
    divided[divided!=0] = np.log(divided[divided!=0])
    return divided


def clarity(photo, parameter):
    # scaling the dimensions -- ask if this size is okay
    scale = max(photo.shape[:2]) / 512.0
    # getting our parameter
    # parameters have to do with pixel diameter for filter, aand color space smoothing
    new_pic = denoise_bilateral(photo,
                                int(32 * scale),
                                50,
                                10 * scale,
                                multichannel=True)

    editted = photo + (photo - new_pic) * parameter
    editted = np.clip(editted, 0, 1)
    return editted


def contrast(photo, parameter):
    mean = photo.mean()
    photo = (photo - mean) * (parameter + 1) + mean
    to_edit = relu(photo)
    editted = 1 - relu(1 - to_edit)
    return editted


def exposure(sig_inv_photo, parameter):
    ''' requires sigmoid_inversed '''
    return sig_inv_photo + parameter * 5


def temp(sig_inv_photo, parameter):
    ''' requires sigmoid_inversed '''
    to_edit = sig_inv_photo
    if parameter > 0:
        to_edit[:, :, 1] += parameter * 1.6
        to_edit[:, :, 0] += parameter * 2
    else:
        to_edit[:, :, 2] -= parameter * 2
        to_edit[:, :, 1] -= parameter * 1
    return to_edit


def tint(sig_inv_photo, parameter):
    ''' requires sigmoid_inversed '''
    to_edit = sig_inv_photo
    to_edit[:, :, 1] -= parameter * 1
    if parameter > 0:
        to_edit[:, :, 0] += parameter * 1
    else:
        to_edit[:, :, 2] -= parameter * 2

    return to_edit


def whites(hsv_photo, parameter):
    ''' requires hsv '''
    white = parameter + 1
    new_values = hsv_photo[:,:,2] + (hsv_photo[:,:,2] * (np.sqrt(white) - 1) * 0.2)
    toReturn = np.dstack([hsv_photo[:,:,0], hsv_photo[:,:,1], new_values])
    return toReturn


def blacks(hsv_photo, parameter):
    ''' requires hsv '''
    black = parameter + 1
    new_values = hsv_photo[:,:,2] + ((1 - hsv_photo[:,:,2]) *
                                 (np.sqrt(black) - 1) * 0.2)
    return np.dstack([hsv_photo[:,:,0], hsv_photo[:,:,1], new_values])


def highlights(hsv_photo, parameter):
    ''' requires hsv '''
    values = hsv_photo[:,:,2]
    highlights_mask = sigmoid(5 * (values - 1))

    return np.dstack([
        hsv_photo[:,:,0], hsv_photo[:,:,1],
        1 - (1 - values) * (1 - highlights_mask * parameter * 5)
    ])


def shadows(hsv_photo, parameter):
    ''' requires hsv '''
    values = hsv_photo[:,:,2]
    shadows_mask = 1 - sigmoid(5 * values)

    return np.dstack([
        hsv_photo[:,:,0], hsv_photo[:,:,1], values * (1 + shadows_mask * parameter * 5)
    ])


def vibrance(hsv_photo, parameter):
    ''' requires hsv '''
    vibrance = parameter + 1
    sat = hsv_photo[:, :, 1]
    vibrance_flag = -sigmoid((sat - 0.5) * 10) + 1

    return np.dstack([
        hsv_photo[:,:,0],
        sat * vibrance * vibrance_flag + sat * (1 - vibrance_flag),
        hsv_photo[:,:,2]
    ])


def saturation(hsv_photo, parameter):
    ''' requires hsv '''
    sat = parameter + 1
    sat_array = hsv_photo[:,:,1]
    sat_array = sat_array * sat
    sat_array = relu(sat_array)
    sat_array = 1 - relu(1 - sat_array)

    return np.dstack([hsv_photo[:,:,0], sat_array, hsv_photo[:,:,2]])


class PhotoEditor():

    # create constants for the photoeditor parameters by adading to nn > hyperparams
    # def __init__(self):


    def __call__(self, photos, parameters):
        # does this list have just one photo?
        editted = np.zeros(np.shape(photos))
        for index, photo in enumerate(photos):
            for param in parameters:
                if (param == exposure) or (param == temp) or (param == tint):  # for exposure, temperature, and tint
                    photoI = param(sigmoid_inversed(photo), parameters[param])
                    photo = sigmoid(photoI)
                elif (param == clarity) or (param == contrast):  # for clarity and contrast
                    photo = param(photo, parameters[param])
                else:  # hsv conversions
                    hsv_edit = param(rgb2hsv(photo), parameters[param])
                    # note: assumes that our pictures have had values from [0, 1]
                    photo = hsv2rgb(hsv_edit)
            # photo = exposure(sigmoid_inversed(photo), parameters[exposure])
            # photo = sigmoid(photo)
            editted[index,:,:,:] = photo
        return editted
