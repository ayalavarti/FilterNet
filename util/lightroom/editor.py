<<<<<<< Updated upstream
import numpy as np
import math
import sys
from skimage.color import rgb2hsv
from matplotlib.colors import hsv_to_rgb  # skimage didn't have one for some reason
import cv2  # pip install opencv-python
=======
import cv2
import numpy as np
from skimage import io, img_as_ubyte, img_as_float32
import math
import sys
from skimage.color import rgb2hsv, hsv2rgb

# from matplotlib.colors import hsv_to_rgb  # skimage didn't have one for some reason
# pip install opencv-python
>>>>>>> Stashed changes


def relu(vector):
    return np.where(vector > 0, vector, 0)


def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))\


def sigmoid_inversed(vector):  # basically the logit function
    # paper implements this differently than I do someone check to see if i need to check for overflow
    return np.log(vector / (1 - vector))


def clarity(photo, parameter):
    # scaling the dimensions -- ask if this size is okay
    scale = max(photo.shape[:2]) / 512.0
<<<<<<< Updated upstream
    # getting our parameter

    # parameters have to do with pixel diameter for filter, aand color space smoothing
    smoothed = cv2.bilateralFilter((photo * 255).astype(np.uint8),
                                   int(32 * scale), 50, 10 * scale) / 255.0

    editted = photo + (photo - smoothed) * parameter
=======
    print(scale)
    # getting our parameter
    # parameters have to do with pixel diameter for filter, aand color space smoothing
    new_pic = np.array(photo.shape)
    cv2.bilateralFilter((photo * 255).astype(np.uint8), new_pic,
                        int(32 * scale), 50, 10 * scale)
    new_pic /= 255.0

    editted = photo + (photo - new_pic) * parameter
>>>>>>> Stashed changes
    return editted


def contrast(photo, parameter):
    mean = photo.mean()
    photo = (photo - mean) * (parameter + 1) + mean
    to_edit = relu(photo)
    editted = 1 - relu(1 - to_edit)
    return editted


def exposure(photo, parameter):
    ''' requires sigmoid_inversed '''
    return sigmoid_inversed(photo) + parameter * 5


def temp(photo, parameter):
    ''' requires sigmoid_inversed '''
<<<<<<< Updated upstream
    to_edit = sigmoid_inversed(photo)
=======
    to_edit = photo
>>>>>>> Stashed changes
    if parameter > 0:
        to_edit[:, :, 1] += parameter * 1.6
        to_edit[:, :, 2] += parameter * 2
    else:
        to_edit[:, :, 0] -= parameter * 2
        to_edit[:, :, 1] -= parameter * 1
    return to_edit


def tint(photo, parameter):
    ''' requires sigmoid_inversed '''
    to_edit = sigmoid_inversed(photo)
    to_edit[:, :, 1] -= parameter * 1
    if parameter > 0:
        to_edit[:, :, 2] += parameter * 1
    else:
        to_edit[:, :, 0] -= parameter * 2

    return to_edit


def whites(hsv_photo, parameter):
    ''' requires hsv '''
    white = parameter + 1
    new_values = hsv_photo[2] + (hsv_photo[2] * (np.sqrt(white) - 1) * 0.2)
    return np.array([hsv_photo[0], hsv_photo[1], new_values])


def blacks(hsv_photo, parameter):
    ''' requires hsv '''
    black = parameter + 1
    new_values = hsv_photo[2] + ((1 - hsv_photo[2]) *
                                 (np.sqrt(black) - 1) * 0.2)
    return np.array([hsv_photo[0], hsv_photo[1], new_values])


def highlights(hsv_photo, parameter):
    ''' requires hsv '''
    values = hsv_photo[2]
    highlights_mask = sigmoid(5 * (values - 1))

    return np.array([
        hsv_photo[0], hsv_photo[1],
        1 - (1 - values) * (1 - highlights_mask * parameter * 5)
    ])


def shadows(hsv_photo, parameter):
    ''' requires hsv '''
    values = hsv_photo[2]
    shadows_mask = 1 - sigmoid(5 * values)

    return np.array([
        hsv_photo[0], hsv_photo[1], values * (1 + shadows_mask * parameter * 5)
    ])


def vibrance(hsv_photo, parameter):
    ''' requires hsv '''
    vibrance = parameter + 1
    sat = hsv_photo[1]
<<<<<<< Updated upstream
    vibrance_flag = -sigmoid((sat - 0.5) * 10) + 1
    return [
        hsv_photo[0],
        sat * vibrance * vibrance_flag + sat * (1 - vibrance_flag),
        hsv_photo[2]
    ]
=======
    print(sat.shape)
    vibrance_flag = -sigmoid((sat - 0.5) * 10) + 1
    print(vibrance_flag.shape)

    return np.array([
        hsv_photo[0],
        sat * vibrance * vibrance_flag + sat * (1 - vibrance_flag),
        hsv_photo[2]
    ])
>>>>>>> Stashed changes


def saturation(hsv_photo, parameter):
    ''' requires hsv '''
    sat = parameter + 1
    sat_array = hsv_photo[1]
    sat_array = sat_array * sat
    sat_array = relu(sat_array)
    sat_array = 1 - relu(1 - sat_array)

    return np.array([hsv_photo[0], sat_array, hsv_photo[2]])


class PhotoEditor():

    # create constants for the photoeditor parameters by adading to nn > hyperparams
    def __init__(self):
        self.edit_funcs = [
            clarity, contrast, exposure, temp, tint, whites, blacks,
            highlights, shadows, vibrance, saturation
        ]

    def __call__(self, photo, parameters):
        # does this list have just one photo?
        for i in range(parameters.size):
            if (i > 1) and (i < 5):  # for exposure, temperature, and tint
                editted = self.edit_funcs[i](sigmoid_inversed(photo),
                                             parameters[i])
                photo = sigmoid_inversed(editted)
            elif (i == 0) or (i == 1):  # for clarity and contrast
                self.edit_funcs[i](photo, parameters[i])
            else:  # hsv conversions
                hsv_edit = self.edit_funcs[i](rgb2hsv(photo), parameters[i])
                # note: assumes that our pictures have had values from [0, 1]
<<<<<<< Updated upstream
                photo = hsv_to_rgb(hsv_edit)

        return photo
=======
                photo = hsv2rgb(hsv_edit)

        return photo


if __name__ == "__main__":
    editor = PhotoEditor()
    test_image = img_as_float32(io.imread("./test.png"))
    # print(test_image[0, 0, 0])
    # hsv_edit = vibrance(rgb2hsv(test_image[:, :, :3]), 0.1)
    # print(rgb2hsv(test_image[:, :, :3]).shape)
    # note: assumes that our pictures have had values from [0, 1]
    photo = clarity(test_image, 0.5)

    io.imsave("./output.png", img_as_ubyte(photo.copy()))
>>>>>>> Stashed changes
