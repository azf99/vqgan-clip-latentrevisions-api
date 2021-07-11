import requests
import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import datetime
import base64
import uuid
import time
from caching_config import *

server = Flask(__name__)

CORS(server)

UPLOAD_FOLDER = "UPLOAD_FOLDER/"

def parse_LatentRevisions(req, key):
    schema = {
        "prompt": "",
        "img": "",
        "w0": 5,
        "text_to_add": "",
        "w1": 0,
        "img_enc": "",
        "w2": 0,
        "ne_img_enc": "",
        "w3": 0
    }
    data = {}
    for i in schema.keys():
        if (i in req.form.keys()) and ("img" not in i):
            data.update({i: req.form[i]})
        elif (i in req.files.keys()) and ("img" in i):
            img = req.files[i]
            filename = save_img("latentrevisions", img, key + i)
            data.update({i + "_path": filename})
    data.update({"id": key})
    return data

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def save_img(label, file, uid):
    #current_time = str(datetime.datetime.now()).replace('-', '_').replace(':', '_')
    filename = UPLOAD_FOLDER + label + uid + ".png"
    file.save(filename)
    return(filename)

@server.route('/index', methods = ["GET", "POST"])
def hello_world():
    return jsonify('Hello, its up!')

@server.route('/styleclip', methods = ["POST"])
def stype_clip():
    k = str(uuid.uuid4())
    prompt = request.form["prompt"]
    # check if reference image is given
    try:
        img = request.files["img"]
        filename = save_img("styleclip", img, k)
    except Exception as e:
        filename = None
        print(e)
        return jsonify({
            "status": "Error. File type not proper."
        }), 400
    #prompt = 'An image with the face of a blonde woman with blonde hair and purple eyes'
    print("Prompt: ", prompt)

    #st = time.time()
    d = {"id": k, "image": filename, "prompt": prompt}
    db.rpush(STYLECLIP_QUEUE, json.dumps(d))

    # keep looping until our model server returns the output
    # predictions
    while True:
        # attempt to grab the output predictions
        output = db.get(k)
        if output is not None:
            # delete the result from the database and break from the polling loop
            db.delete(k)
            break
        # sleep for a small amount to give the model a chance to process the input image
        time.sleep(CLIENT_SLEEP)
    res = open(output, "rb")
    encoded_image = base64.b64encode(res.read()).decode("utf-8")

    #print("Time Taken:", time.time() - st)
    # return the generated image
    return jsonify({
        "status": "success",
        "image": encoded_image
    })

@server.route('/latent_revision', methods = ["POST"])
def latent_revisions():
    k = str(uuid.uuid4())
    d = parse_LatentRevisions(request, k)
    #prompt = 'An image with the face of a blonde woman with blonde hair and purple eyes'
    print("Prompt: ", d["prompt"])

    #st = time.time()
    #d = {"id": k, "image": filename, "prompt": prompt}
    db.rpush(LATENTREVISIONS_QUEUE, json.dumps(d))

    # keep looping until our model server returns the output
    # predictions
    while True:
        # attempt to grab the output predictions
        output = db.get(k)
        if output is not None:
            # delete the result from the database and break from the polling loop
            db.delete(k)
            break
        # sleep for a small amount to give the model a chance to process the input image
        time.sleep(CLIENT_SLEEP)
    res = open(output, "rb")
    encoded_image = base64.b64encode(res.read()).decode("utf-8")

    #print("Time Taken:", time.time() - st)
    # return the generated image
    return jsonify({
        "status": "success",
        "image": encoded_image
    })

server.run(HOST, port = PORT, threaded = THREADED, debug = DEBUG)