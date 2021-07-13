from StyleCLIP.StyleCLIP import *
from LatentRevisions.LatentRevisions import *
from caching_config import *
import time
import json
from threading import Thread

def StyleCLIP_Thread():
        # continually pool for new images to process
        while True:
            # attempt to grab an image from the database
            req = db.lpop(STYLECLIP_QUEUE)
            if req == None:
                time.sleep(SERVER_SLEEP)
                continue
            q = json.loads(db.get(req))

            db.set(req, json.dumps({"status": "processing", "steps": 0}))

            model = StyleCLIP(prompt = q["prompt"], img_path = q["image_path"], uid = req)
            out_path = model.run()
            print("Processed id: ", req, " Saved at out_path: ", out_path)

            out = {"out_path": out_path}
            db.set(req, json.dumps(out))

def LatentRevisions_Thread():
        # continually pool for new images to process
        while True:
            # attempt to grab an image from the database
            req = db.lpop(LATENTREVISIONS_QUEUE)
            if req == None:
                time.sleep(SERVER_SLEEP)
                continue

            q = json.loads(db.get(req))

            db.set(req, json.dumps({"status": "processing", "steps": 0}))
            q.pop("type")
            q.pop("id")

            model = LatentRevisions(**q)
            #out_path = model.run()
            while model.itt <= model.iter_limit:
                model.train(model.itt)
                model.itt += 1
                db.set(req, json.dumps({"status": "processing", "steps": model.itt}))
            out_path = model.output_path
            print("Processed id: ", req, " Saved at out_path: ", out_path)

            out = {"out_path": out_path}
            db.set(req, json.dumps(out))

if __name__ == "__main__":
    lv = []
    tsc = []
    for i in range(STYLECLIP_THREADS):
        t = Thread(target = StyleCLIP_Thread)
        t.start()
        tsc.append(t)
    print("Started StyleCLIP")

    for i in range(LATENTREVISIONS_THREADS):
        t = Thread(target = LatentRevisions_Thread)
        t.start()
        lv.append(t)
    print("Started LatenRevisions....")
    for i in tsc:
        i.join()

    for i in lv:
        i.join()
