from StyleCLIP.StyleCLIP import *
from LatentRevisions.LatentRevisions import *
from caching_config import *
import time
from threading import Thread

def StyleCLP_Thread():
    	# continually pool for new images to process
	while True:
		# attempt to grab an image from the database
		req = db.lpop(STYLECLIP_QUEUE, 0)
        if req is None:
            time.sleep(1.0)
            continue

		q = json.loads(req)

        model = StyleCLIP(prompt = q["prompt"], img_path = q["image"])
        out_path = model.run()
        print("Processed id: ", d["id"], " Saved at out_path: ", out_path)
        db.set(q["id"], out_path)

def LatentRevisions_Thread():
    	# continually pool for new images to process
	while True:
		# attempt to grab an image from the database
		req = db.lpop(STYLECLIP_QUEUE, 0)
        if req is None:
            time.sleep(1.0)
            continue

		q = json.loads(req)

        model = LatentRevisions(prompt = q["prompt"])
        out_path = model.run()
        print("Processed id: ", d["id"], " Saved at out_path: ", out_path)
        db.set(q["id"], out_path)

if __name__ == "__main__":
    lv = []
    tsc = []
    for i in range(NUM_THREADS):
        t = Thread(target = StyleCLP_Thread)
        t.start()
        tsc.append(t)
    
    for i in range(NUM_THREADS):
        t = Thread(target = LatentRevisions_Thread)
        t.start()
        lv.append(t)
    
    for i in tsc:
        i.join()
    
    for i in lv:
        i.join()