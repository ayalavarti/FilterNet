import os
import logging

from PIL import Image
from util.sys import edit_original
from util.query import DatastoreQuery
from flask import Flask, render_template, request, jsonify
from skimage.io import imsave

from nn.models import *

template_dir = os.path.abspath('./web/templates')
static_dir = os.path.abspath('./web/static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.environ["FLASK_KEY"]


def init_model():
    generator = Generator()
    generator.load_weights(os.environ["MODEL_PATH"])
    logging.info("Weights loaded")
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

    edit = edit_original(image, generator)
    id = query.increment_latest_id()
    filepath = static_dir + "/images/" + str(id) + "-edit.png"
    imsave(filepath, edit)

    filepath = "static/images/" + str(id) + "-edit.png"
    return jsonify({'status': "Success", "image_url": filepath})
