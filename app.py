import io
import os
import tempfile

import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify

from db import *
from nn.models import *

MODEL_ID = os.environ["MODEL_ID"]
dbURI = os.environ["DATABASE_URL"]

template_dir = os.path.abspath('./web/templates')
static_dir = os.path.abspath('./web/static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.environ["FLASK_KEY"]


def init_model():
    generator = Generator()
    discriminator = Discriminator()

    query = FilterNetQuery(dbURI)
    blob, ret = query.read_model(MODEL_ID)

    if ret is None or not ret or blob is None:
        print("Error reading from database")

    if blob:    
        with tempfile.NamedTemporaryFile(suffix='.h5') as gen_file, \
            tempfile.NamedTemporaryFile(suffix='.h5') as disc_file:
            gen_file.write(blob[0])
            disc_file.write(blob[1])

            generator.load_weights(gen_file.name)
            discriminator.load_weights(disc_file.name)


init_model()


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
