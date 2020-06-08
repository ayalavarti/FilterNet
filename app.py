from flask import Flask, render_template, request
import os
from db.query import DatabaseQuery

template_dir = os.path.abspath('./web/templates')
static_dir = os.path.abspath('./web/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

dbURI = os.environ["DATABASE_URL"]
# dbQuery = DatabaseQuery()


@app.route('/')
def init_app():
    return render_template('home.html')
