import tempfile

from flask import Flask, render_template, request, send_file, jsonify
from matplotlib import pyplot as plt
import os
from db.query import FilterNetQuery
from nn.models import *
import numpy as np
import cv2
import io
from PIL import Image

MODEL_ID = 2

template_dir = os.path.abspath('./web/templates')
static_dir = os.path.abspath('./web/static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = "super secret key"

generator = Generator()
discriminator = Discriminator()

dbURI = os.environ["DATABASE_URL"]

query = FilterNetQuery(dbURI)
blob, ret = query.read_model(MODEL_ID)

if ret is None or not ret:
    exit(1)

with tempfile.NamedTemporaryFile(suffix='.h5') as gen_file, \
        tempfile.NamedTemporaryFile(suffix='.h5') as disc_file:
    gen_file.write(blob[0])
    disc_file.write(blob[1])

    generator.load_weights(gen_file.name)
    discriminator.load_weights(disc_file.name)


def decode_image(file):
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return imageRGB


@app.route('/')
def init_app():
    return render_template('home.html')


@app.route("/edit", methods=['POST'])
def edit_photo():
    file = request.files['file'].read()
    image = decode_image(file)

    if 'id' in request.form:
        print(request.form['id'])
        # Edit image
        res = {"status": "Success", "id": 1}
    else:
        im = Image.fromarray(image)

        file_object = io.BytesIO()
        im.save(file_object, 'PNG')

        file_object.seek(0)

        res = {"status": "Success", "id": 1}

    return jsonify(res)
    # return send_file(file_object, mimetype='image/PNG')
