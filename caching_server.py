import requests
import json

from flask import Flask, jsonify, request
from flask_cors import CORS
import datetime
import base64
import uuid
import time
import caching_config import *

server = Flask(__name__)

CORS(server)

UPLOAD_FOLDER = "UPLOAD_FOLDER/"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def save_img(label, file):
    #current_time = str(datetime.datetime.now()).replace('-', '_').replace(':', '_')
    filename = UPLOAD_FOLDER + label + uuid.uuid4() + ".png"
    file.save(filename)
    return(filename)

@server.route('/index', methods = ["GET", "POST"])
def hello_world():
    return jsonify('Hello, its up!')

@server.route('/styleclip', methods = ["POST"])
def stype_clip():
    prompt = request.form["prompt"]
    # check if reference image is given
    try:
        img = request.files["upload_file"]
        filename = save_img("styleclip", img)
    except Exception as e:
        filename = None
        print(e)
        return jsonify({
            "status": "Error. File type not proper."
        }), 400
    #prompt = 'An image with the face of a blonde woman with blonde hair and purple eyes'
    print("Prompt: ", prompt)

    #st = time.time()
    k = str(uuid.uuid4())
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
    """
    TODO: Add other images and weights
    """
    prompt = request.form["prompt"]
    # check if reference image is given
    try:
        img = request.files["upload_file"]
        filename = save_img("latentrevisions", img)
    except Exception as e:
        filename = None
        print(e)
        return jsonify({
            "status": "Error. File type not proper."
        }), 400
    #prompt = 'An image with the face of a blonde woman with blonde hair and purple eyes'
    print("Prompt: ", prompt)

    #st = time.time()
    k = str(uuid.uuid4())
	d = {"id": k, "image": filename, "prompt": prompt}
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