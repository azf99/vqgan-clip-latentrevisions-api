import requests
path = "http://a0e1ca37de9b.ngrok.io/latent_revision"
#path = "http://34.205.133.166/styleclip"
d = {
     "prompt": 'A beautiful person',
      "w0": 5,
       "text_to_add": "queen looking at us",
       "w1": 2,
       "w2": 5,
       "w3": 1,
}

imgs = {
    "img": "architecture.jpeg",  
    "img_enc": "test/test.jpeg",
    "ne_img_enc": "architecture.jpeg"
     }

for key, value in imgs.items():
    imgs.update({key: open(value, "rb").read()})
    
res = requests.post(path, data = d, files = imgs)
print(res.json())

i = res.json()["id"]

res = requests.post("http://34.205.133.166/check", data = {"id": i})
print(res.json().keys())
print(res.json()["rank"])

'''
{'prompt': 'A beautiful person',
 'img_path': 'UPLOAD_FOLDER/latentrevisionsc3b067b4-f815-4c6f-9be4-84d9a38fac02img.png',
 'w0': '5',
 'img_enc_path': 'UPLOAD_FOLDER/latentrevisionsc3b067b4-f815-4c6f-9be4-84d9a38fac02img_enc.png',
 'w2': '5',
 'ne_img_enc_path': 'UPLOAD_FOLDER/latentrevisionsc3b067b4-f815-4c6f-9be4-84d9a38fac02ne_img_enc.png',
 'w3': '1'}
'''