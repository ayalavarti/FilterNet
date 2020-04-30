import matplotlib
from matplotlib import pyplot as plt
from skimage import io


def visualize_image(image):
	io.imshow(image)
	io.show()


def visualize_batch(batch, model_edits, display, num_display):
	"""
	Method to be used for visualizing output images
	"""
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
