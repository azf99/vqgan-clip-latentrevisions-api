import requests
import json

path = "http://100.24.21.46/styleclip"

d = {
     "prompt": "An image with the face of a blonde woman with blonde hair and purple eyes"
}

input_image = "test/test.jpeg"

res = requests.post(path, data = d, files = {"img": open(input_image, "rb").read()})

from threading import Thread
import time
results = []

def make_call(data):
    st = time.time()
    res = requests.post(path, data = d, files = {"img": open(input_image, "rb").read()})
    results.append(res.json())
    print("Received", time.time() - st)

NUM_THREADS = 5

t = []

st = time.time()
for i in range(NUM_THREADS):
    t1 = Thread(target = make_call, args = (d))
    t.append(t1)
    t1.start()

for i in range(NUM_THREADS):
    t[i].join()
print("Time Taken:", time.time() - st)


import base64
from PIL import Image
import io
image = base64.b64decode(results[0]["image"])       
fileName = 'test.jpeg'

imagePath = (fileName)
img = Image.open(io.BytesIO(image))
img.save(imagePath, 'jpeg')


