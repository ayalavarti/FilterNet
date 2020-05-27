from flask import Flask, render_template, request
import os


template_dir = os.path.abspath('./web/templates')
static_dir = os.path.abspath('./web/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

os.environ[]
print("test")


@app.route('/')
def init_app():
    return render_template('home.html')
