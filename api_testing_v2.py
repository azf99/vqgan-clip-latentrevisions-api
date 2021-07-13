import requests
import json

# don't use https
ip = '44.193.226.152'
#path = "http://{}/styleclip".format(ip)
path = "http://{}/latent_revision".format(ip)
'''
weighting range, suggested from collab, slider:
-5 to 5 but could put in any number we want
tried with bigger weights, makes it slower a little bit, 
matrix operations by bigger number.  Very big number will effect it.
'''
'''
d = {
     "prompt": 'A beautiful person', #w0
      "w0": 5,
       "text_to_add": "queen looking at us", #w1
       "w1": 2,
       "w2": 5,
       "w3": 1,
}
# remove key from dictionary because on line 21 it's going to look for it
# prompt is absolutely needed
imgs = {
    "img": "architecture.jpg",  #no weight, starter image
    "img_enc": "test/test.jpeg", #w2
#    "ne_img_enc": "" #w3
     }

for key, value in imgs.items():
    imgs.update({key: open(value, "rb").read()})
    
res = requests.post(path, data = d, files = imgs)
print(res.json())
i = res.json()["id"]
'''



# run check endpoint
# second step after the request
# if the image is already in processing it returns 0
# if the image is processed it will send you the image itself, not the rank
# outputted successful image cannot be retrieved from redis once it's sent over once
# all the stored images are kept on hard disk
res = requests.post("http://{}/check".format(ip), data = {"id": '07f918d7-2bed-4f66-9c0e-8bcc922cd32a'})
print(res.json().keys())

if(res.json().get('rank') != None):
    print(res.json()["rank"])

if(res.json().get('image') != None):
    import base64
    from PIL import Image
    import io
    image = base64.b64decode(res.json()['image'])       
    fileName = 'final-output-test.png'

    imagePath = (fileName)
    img = Image.open(io.BytesIO(image))
    img.save(imagePath, 'png')

'''
{'prompt': 'A beautiful person',
 'img_path': 'UPLOAD_FOLDER/latentrevisionsc3b067b4-f815-4c6f-9be4-84d9a38fac02img.png',
 'w0': '5',
 'img_enc_path': 'UPLOAD_FOLDER/latentrevisionsc3b067b4-f815-4c6f-9be4-84d9a38fac02img_enc.png',
 'w2': '5',
 'ne_img_enc_path': 'UPLOAD_FOLDER/latentrevisionsc3b067b4-f815-4c6f-9be4-84d9a38fac02ne_img_enc.png',
 'w3': '1'}
'''