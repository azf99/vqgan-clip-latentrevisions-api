from StyleCLIP.StyleCLIP import *
from LatentRevisions import *
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/index', methods = ["GET", "POST"])
def hello_world():
    return jsonify('Hello, its up!')

@app.route('/styleclip', methods = ["POST"])
def stype_clip():
    #prompt = request.form["prompt"]
    prompt = 'An image with the face of a blonde woman with blonde hair and purple eyes'
    model = StyleCLIP(prompt = prompt)
    path = model.run()
    return jsonify({
        "status": "success",
        "image": path
    })

@app.route('/latent_revision', methods = ["POST"])
def latent_revisions():
    #prompt = request.form["prompt"]
    prompt = 'A beautiful person'
    model = LatentRevisions(prompt = prompt)
    path = model.run()
    return jsonify({
        "status": "success",
        "image": path
    })