import redis

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_CHANS = 3
# initialize constants used for server queuing
STYLECLIP_QUEUE = "styleclip_queue"
LATENTREVISIONS_QUEUE = "latentrevisions_queue"
SERVER_SLEEP = 1
CLIENT_SLEEP = 2

db = redis.StrictRedis(host="localhost", port=6379, db=0)

STYLECLIP_THREADS = 3
LATENTREVISIONS_THREADS = 1

# For Flask Server

HOST = "0.0.0.0"
PORT = 80
THREADED = True
DEBUG = False
