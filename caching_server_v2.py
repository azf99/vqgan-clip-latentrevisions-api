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
    data = {
        "prompt": "",
        "output_path": "./latentrevisions_out",
        "img": '',
        "w0": 5,
        "text_to_add": "",
        "w1": 0,
        "img_enc": "",
        "w2": 0,
        "ne_img_enc": "",
        "w3": 0
    }
    for i in data.keys():
        if (i in req.form.keys()) and ("img" not in i):
            data.update(i, req.form[i])
        elif (i in req.files.keys()) and ("img" in i):
            img = req.files[i]
            filename = save_img("latentrevisions", img, key + i)
            data.pop(i)
            data.update(i + "_path", filename)
    data.update("id", key)
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
        img = request.files["upload_file"]
        filename = save_img("styleclip", img, k)
    except Exception as e:
        filename = None
        print(e)
        return jsonify({
            "status": "error",
            "message": "file type not proper"
        }), 400
    #prompt = 'An image with the face of a blonde woman with blonde hair and purple eyes'
    print("Prompt: ", prompt)

    #st = time.time()
    d = {"image_path": filename, "prompt": prompt, "type": "sc"}
    db.set(k, json.dumps(d))

    db.rpush(STYLECLIP_QUEUE, k)

    #print("Time Taken:", time.time() - st)
    # return the generated image
    return jsonify({
        "status": "success",
        "message": "upload successful",
        "id": k
    })

@server.route('/latent_revision', methods = ["POST"])
def latent_revisions():
    """
    TODO: Add other images and weights
    """
    k = str(uuid.uuid4())
    d = parse_LatentRevisions(request, key)
    #prompt = 'An image with the face of a blonde woman with blonde hair and purple eyes'
    print("Prompt: ", d["prompt"])

    #d = {"image_path": filename, "prompt": prompt, "type": "lr"}
    d.update("type", "lr")
    db.set(k, json.dumps(d))

    #st = time.time()
    db.rpush(LATENTREVISIONS_QUEUE, k)
    #print("Time Taken:", time.time() - st)
    # return the generated image
    return jsonify({
        "status": "success",
        "message": "upload successful",
        "id": k
    })

@server.route('/check', methods = ["POST"])
def check():
    """
    check rank and get result
    """
    key = request.form["id"]

    item = json.loads(db.get(key))

    if item == None:
        return jsonify({
            "status": "error",
            "message": "is does not exist"
        })
    else:
        if "out_path" in item.keys():
            res = open(item["out_path"], "rb")
            encoded_image = base64.b64encode(res.read()).decode("utf-8")
            return jsonify({
                    "status": "success",
                    "image": encoded_image
                })
        elif "prompt" in item.keys():
            pos = 0
            if item["type"] == "lr":
                pos = db.lpos(LATENTREVISIONS_QUEUE, key)
            else:
                pos = db.lpos(STYLECLIP_QUEUE, key)
            return jsonify({
                "status": "processing",
                "id": key,
                "rank": pos
            })
        elif "status" in item.keys():
            return jsonify({
                "status": "processing",
                "id": key,
                "rank": 0
            })
    

server.run(HOST, port = PORT, threaded = THREADED, debug = DEBUG)