# mead-backend

NOTE: All this should be done under the `pytorch_latest_p37` environment on the EC2 Instance.
It can be activated by running `conda activate pytorch_latest_p37`

### Prerequisites
Make sure the installed version of `redis` is `6.2.4` and upgrade to this if any older version is installed.

### Instructions for Installation
1. Clone this repository.
2. `cd mead-backend`
3. Install all the dependencies and download all the required weights: `sudo sh setup.sh`
4. Start the redis in-memory database using: `redis-server`

### Using the base classes
```python
# For LatentRevisions
from LatentRevisions.LatentRevisions import LatentRevisions
d = {
    "prompt": 'A beautiful person',
    "w0": 5,
    "img_path": "architecture.jpeg",  
    "img_enc_path": "test/test.jpeg",
    "ne_img_enc_path": "architecture.jpeg",
    "text_to_add": "queen looking at us", 
    "w1": 2,        
    "w2": 5,  
    "w3": 1,
}
model = LatentRevisions(**d)
path_to_output_image = model.run()

# Running with just the prompt
p = "A beautiful person"
model = LatentRevisions(prompt = p)
path_to_output_image = model.run()

# For StyleCLIP
from StyleCLIP.StyleCLIP import StyleCLIP

p = "A girl with purple eyes"
filename = "test/test.jpeg"
model = StyleCLIP(prompt = p, img_path = filename)
path_to_output_image = model.run()
```

### Files
* [`app.py`](app.py): This file has the basic API's for testing. Its not a threaded application and will only process one request at a time.
* [`inference_server.py`](inference_server.py): This file runs threads for inferencing from images from the redis queue. This runs multiple threads for both StyleCLIP and LatentRevisions. The number of threads can be defined in the [`caching_config.py`](caching_config.py). Max possible thhreads without causing CUDA OOM errors on a single Tesla T4 are already defined in the file.
* [`caching_server.py`](caching_server.py): Run this server after running [`inference_server.py`](inference_server.py). This is also a Flask app with endpoints for both the models. This is a threaded application and pushes all the incoming images in the Redis queue.

### Input POST request format for LatentRevisions
The request should have these in the form attributes:
accessible using `request.form` in `flask`
```json
{
"prompt": "",   # mandatory
"w0": 5,        #weight for prompt

# OPTIONAL
"text_to_add": "", 
"w1": 0,        # weight for "text_to_add"
"w2": 0,    weight for "img_enc"
"w3": 0     # weight "ne_img_enc"
}

FILES(OPTIONAL) accessible using request.files in flask: 
"img": file,      # starter image
"img_enc": file,  # image
"ne_img_enc": file, # negative image
```
### Architecture
![Architecture](architecture.jpeg)

### Performance
A single process takes up 90%+ processing power of the GPU, so it is recommended to run only one process at a time. Running multiple threads at the same time results in delay for individual threads, because the GPU usage is too much. Still multiple threads are possible because of extra GPU memory available.
* LatentRevision: Processing takes 45-50 seconds for a 100 iterations for a single call(single threaded). Multithreaded would be slower for individual threads, but still faster that processing 'n' images in a single thread.
* StyleCLIP: The model is run for 50 iterations, which takes 8-10 seconds to finish per call. First inference is always 20+ seconds.
