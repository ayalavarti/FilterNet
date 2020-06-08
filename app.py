import tempfile

from flask import Flask, render_template, request
import os
from db.query import FilterNetQuery
from nn.models import *

template_dir = os.path.abspath('./web/templates')
static_dir = os.path.abspath('./web/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

generator = Generator()
discriminator = Discriminator()

dbURI = os.environ["DATABASE_URL"]

query = FilterNetQuery(dbURI)
blob, ret = query.read_model(1)

if ret is None or not ret:
    exit(1)

with tempfile.NamedTemporaryFile(suffix='.h5') as gen_file, \
        tempfile.NamedTemporaryFile(suffix='.h5') as disc_file:
    gen_file.write(blob[0])
    disc_file.write(blob[1])

    generator.load_weights(gen_file.name)
    discriminator.load_weights(disc_file.name)


@app.route('/')
def init_app():
    return render_template('home.html')
