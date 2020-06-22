import os
import logging
import base64

from io import BytesIO
from PIL import Image
from util.sys import edit_original
from util.query import DatastoreQuery
from flask import Flask, render_template, request, jsonify
from skimage.io import imsave

from nn.models import *

#template_dir = os.path.abspath('./web/templates')
#static_dir = os.path.abspath('./web/static')
static_dir = os.path.abspath('./static')

app = Flask(__name__)
#app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.environ["FLASK_KEY"]


def init_model():
    generator = Generator()
    discriminator = Discriminator()

    # Set up tf checkpoint manager
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator)

    manager = tf.train.CheckpointManager(
        checkpoint, os.environ["MODEL_PATH"], max_to_keep=3)

    # Restores the latest checkpoint using from the manager
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    logging.info("Weights lo aded")
    return generator


generator = init_model()
query = DatastoreQuery()


def decode_image(file):
    img = Image.open(file)
    imageRGB = np.array(img)
    return imageRGB


@app.route('/')
def init_app():
    return render_template('home.html')


@app.route("/edit", methods=['POST'])
def edit_photo():
    file = request.files['file']
    image = decode_image(file)
    logging.info("Image received")

    edit = edit_original(image, generator) * 255
    id = query.increment_latest_id()

    data = BytesIO()
    img = Image.fromarray(edit.astype(np.uint8))
    img.save(data, "JPEG")
    data64 = base64.b64encode(data.getvalue())

    img_binary = u'data:img/jpeg;base64,'+ data64.decode('utf-8')

    return jsonify({'status': "Success", "image": img_binary, "id": id})
