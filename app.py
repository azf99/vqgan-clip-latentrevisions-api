from StyleCLIP.StyleCLIP import *
from LatentRevisions.LatentRevisions import *
from flask import Flask, jsonify, request
from flask_cors import CORS
import datetime
import base64
import time

app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = "UPLOAD_FOLDER/"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def save_img(label, file):
    current_time = str(datetime.datetime.now()).replace('-', '_').replace(':', '_')
    filename = UPLOAD_FOLDER + label + current_time + ".png"
    file.save(filename)
    return(filename)

@app.route('/index', methods = ["GET", "POST"])
def hello_world():
    return jsonify('Hello, its up!')

@app.route('/styleclip', methods = ["POST"])
def stype_clip():
    prompt = request.form["prompt"]
    # check if reference image is given
    try:
        img = request.files["upload_file"]
        filename = save_img("styleclip", img)
    except Exception as e:
        filename = None
        print(e)
    #prompt = 'An image with the face of a blonde woman with blonde hair and purple eyes'
    print("Prompt: ", prompt)

    st = time.time()
    model = StyleCLIP(prompt = prompt, img_path = filename)
    path = model.run()

    res = open(path, "rb")
    encoded_image = base64.b64encode(res.read()).decode("utf-8")

    del model

    print("Time Taken:", time.time() - st)
    # return the generated image
    return jsonify({
        "status": "success",
        "image": encoded_image
    })

@app.route('/latent_revision', methods = ["POST"])
def latent_revisions():
    prompt = request.form["prompt"]
    # check if reference image is given
    try:
        img = request.files["upload_file"]
        filename = save_img("latent_revisions", img)
    except Exception as e:
        filename = None
        print(e)
    #prompt = 'An image with the face of a blonde woman with blonde hair and purple eyes'
    print("Prompt: ", prompt)

    st = time.time()
    #prompt = 'A beautiful person'
    model = LatentRevisions(prompt = prompt)
    path = model.run()
    
    res = open(path, "rb")
    encoded_image = base64.b64encode(res.read()).decode("utf-8")

    del model

    print("Time Taken:", time.time() - st)
    # return the generated image
    return jsonify({
        "status": "success",
        "image": encoded_image
    })

import app_config

app.run(app_config.HOST, port = app_config.PORT, threaded = app_config.THREADED, debug = app_config.DEBUG)
