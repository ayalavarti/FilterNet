import matplotlib
import numpy as np
from scipy import special
from skimage.color import rgb2hsv, hsv2rgb
from skimage.restoration import denoise_bilateral
from matplotlib import pyplot as plt

import nn.hyperparameters as hp


# ======== Pre-, post- processing functions ========
def identity(vector):
    return vector


def relu(vector):
    vector[vector < 0] = 0
    return vector


def sigmoid(vector):
    return special.expit(vector)


def sigmoid_inverse(vector):
    denom = 1 - vector
    denom[denom < .001] = .001
    divided = vector / denom
    divided[divided <= 0] = 0
    divided[divided != 0] = np.log(divided[divided != 0])
    return divided


# Method to be used for visualizing output images
def visualize_batch(batch, model_edits, display, num_display):
	if not display:
		matplotlib.use('Agg')
	num_images = min(batch.shape[0], num_display)
	fig, axs = plt.subplots(nrows=3, ncols=num_images)
	fig.suptitle("Images\n ")
	for ind, ax in enumerate(axs):
		for i in range(len(ax)):
			a = ax[i]
			if ind == 0:
				a.imshow(batch[i, 0], cmap="Greys")
				a.set(title="Unedited")
			elif ind == 1:
				a.imshow(model_edits[i], cmap="Greys")
				a.set(title="Model")
			else:
				a.imshow(batch[i, 1], cmap="Greys")
				a.set(title="Expert")
			plt.setp(a.get_xticklabels(), visible=False)
			plt.setp(a.get_yticklabels(), visible=False)
			a.tick_params(axis='both', which='both', length=0)
	if display:
		plt.show()
	else:
		plt.savefig('output.jpg', bbox_inches='tight')

# Method to calculate PSNR between to images as a performance metric
def PSNR(model, expert):
	mean_squared_error = np.mean((expert - model) ** 2)
	if (mean_squared_error == 0):
		return 100
	max_pixel_val = 1.0
	return 20 * np.log10(max_pixel_val / np.sqrt(mean_squared_error))


# ======== Custom Lightroom Filters ========
class Filter:
    def __init__(self, pre_process, f, post_process):
        self.pre_process = pre_process
        self.filter = f
        self.post_process = post_process

    def edit_image(self, photo, parameter):
        photo = self.pre_process(photo)
        photo = self.filter(photo, parameter)
        out_photo = self.post_process(photo)
        return out_photo


class Clarity(Filter):
    def __init__(self):
        super(Clarity, self).__init__(identity, self._filter, identity)

    @staticmethod
    def _filter(photo, parameter):
        # scale is 1/8 of the maximum image size
        scale = max(photo.shape[:2]) / (8 * max(photo.shape[:2]))
        # parameters have to do with pixel diameter for filter,
        # and color space smoothing
        new_pic = denoise_bilateral(photo,
                                    int(32 * scale),
                                    50,
                                    10 * scale,
                                    multichannel=True)

        edited = photo + (photo - new_pic) * parameter
        edited = np.clip(edited, 0, 1)
        return edited

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Contrast(Filter):
    def __init__(self):
        super(Contrast, self).__init__(identity, self._filter, identity)

    @staticmethod
    def _filter(photo, parameter):
        mean = photo.mean()
        photo = (photo - mean) * (parameter + 1) + mean
        to_edit = relu(photo)
        edited = 1 - relu(1 - to_edit)
        return edited

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Exposure(Filter):
    def __init__(self):
        # Requires sigmoid inverse
        super(Exposure, self).__init__(sigmoid_inverse, self._filter, sigmoid)

    @staticmethod
    def _filter(sig_inv_photo, parameter):
        return sig_inv_photo + parameter * 5

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Temp(Filter):
    def __init__(self):
        # Requires sigmoid inverse
        super(Temp, self).__init__(sigmoid_inverse, self._filter, sigmoid)

    @staticmethod
    def _filter(sig_inv_photo, parameter):
        to_edit = sig_inv_photo
        if parameter > 0:
            to_edit[:, :, 1] += parameter * 1.6
            to_edit[:, :, 0] += parameter * 2
        else:
            to_edit[:, :, 2] -= parameter * 2
            to_edit[:, :, 1] -= parameter * 1
        return to_edit

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Tint(Filter):
    def __init__(self):
        # Requires sigmoid inverse
        super(Tint, self).__init__(sigmoid_inverse, self._filter, sigmoid)

    @staticmethod
    def _filter(sig_inv_photo, parameter):
        to_edit = sig_inv_photo
        to_edit[:, :, 1] -= parameter * 1
        if parameter > 0:
            to_edit[:, :, 0] += parameter * 1
        else:
            to_edit[:, :, 2] -= parameter * 2
        return to_edit

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Whites(Filter):
    def __init__(self):
        # Requires HSV
        super(Whites, self).__init__(rgb2hsv, self._filter, hsv2rgb)

    @staticmethod
    def _filter(hsv_photo, parameter):
        white = parameter + 1
        white = 0 if white < 0 else white
        new_values = hsv_photo[:, :, 2] + (hsv_photo[:, :, 2] *
                                           (np.sqrt(white) - 1) * 0.2)
        toReturn = np.dstack(
            [hsv_photo[:, :, 0], hsv_photo[:, :, 1], new_values])
        return toReturn

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Blacks(Filter):
    def __init__(self):
        # Requires HSV
        super(Blacks, self).__init__(rgb2hsv, self._filter, hsv2rgb)

    @staticmethod
    def _filter(hsv_photo, parameter):
        black = parameter + 1
        black = 0 if black < 0 else black
        new_values = hsv_photo[:, :, 2] + ((1 - hsv_photo[:, :, 2]) *
                                           (np.sqrt(black) - 1) * 0.2)
        return np.dstack([hsv_photo[:, :, 0], hsv_photo[:, :, 1], new_values])

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Highlights(Filter):
    def __init__(self):
        # Requires HSV
        super(Highlights, self).__init__(rgb2hsv, self._filter, hsv2rgb)

    @staticmethod
    def _filter(hsv_photo, parameter):
        values = hsv_photo[:, :, 2]
        highlights_mask = sigmoid(5 * (values - 1))

        return np.dstack([
            hsv_photo[:, :, 0], hsv_photo[:, :, 1],
            1 - (1 - values) * (1 - highlights_mask * parameter * 5)
        ])

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Shadows(Filter):
    def __init__(self):
        # Requires HSV
        super(Shadows, self).__init__(rgb2hsv, self._filter, hsv2rgb)

    @staticmethod
    def _filter(hsv_photo, parameter):
        values = hsv_photo[:, :, 2]
        shadows_mask = 1 - sigmoid(5 * values)

        return np.dstack([
            hsv_photo[:, :, 0], hsv_photo[:, :, 1],
            values * (1 + shadows_mask * parameter * 5)
        ])

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Vibrance(Filter):
    def __init__(self):
        # Requires HSV
        super(Vibrance, self).__init__(rgb2hsv, self._filter, hsv2rgb)

    @staticmethod
    def _filter(hsv_photo, parameter):
        vibrance = parameter + 1
        sat = hsv_photo[:, :, 1]
        vibrance_flag = -sigmoid((sat - 0.5) * 10) + 1

        return np.dstack([
            hsv_photo[:, :, 0],
            sat * vibrance * vibrance_flag + sat * (1 - vibrance_flag),
            hsv_photo[:, :, 2]
        ])

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class Saturation(Filter):
    def __init__(self):
        # Requires HSV
        super(Saturation, self).__init__(rgb2hsv, self._filter, hsv2rgb)

    @staticmethod
    def _filter(hsv_photo, parameter):
        sat = parameter + 1
        sat_array = hsv_photo[:, :, 1]
        sat_array = sat_array * sat
        sat_array = relu(sat_array)
        sat_array = 1 - relu(1 - sat_array)

        return np.dstack([hsv_photo[:, :, 0], sat_array, hsv_photo[:, :, 2]])

    def __call__(self, photo, parameter):
        return self.edit_image(photo, parameter)


class PhotoEditor:
    filters = [
        Clarity(),
        Contrast(),
        Exposure(),
        Temp(),
        Tint(),
        Whites(),
        Blacks(),
        Highlights(),
        Shadows(),
        Vibrance(),
        Saturation()
    ]

    @classmethod
    def edit(cls, photos, parameters, ind=None):
        """
		Static class method that will edit a batch of images given a batch of
		:parameters. Applies each of the K filters for the ith image using the
		ith set of K parameters. Each parameter should be between [-1, 1].

		Can optionally include a list of indices indicate specific filters to
		apply. However, parameters adn photos should still be of the required
		shape (detailed below).

		Order of filters reflects order of parameters:
		[clarity, contrast, exposure, temp, tint, whites, blacks, highlights,
		shadows, vibrance, saturation]

		:param photos: numpy array of shape
					[hp.batch_size, hp.img_size, hp.img_size, 3]
		:param parameters: numpy array of shape
					[hp.batch_size, hp.K]
		:param ind: list of indices to apply filters of
		:return:
		"""
        if parameters.shape[1] != hp.K:
            raise ValueError("Incorrect number of filter parameters found")
        edited = np.zeros(np.shape(photos))
        for i in range(len(photos)):
            photo, params = photos[i], parameters[i]
            new_photo = np.copy(photo)
            if ind:
                app_f, app_params = np.take(cls.filters,
                                            ind), np.take(params, ind)
            else:
                app_f, app_params = cls.filters, params

            for fltr, p in zip(app_f, app_params):
                new_photo = fltr(new_photo, p)
            edited[i] = new_photo
        return edited
