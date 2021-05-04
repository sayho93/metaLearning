import tensorflow as tf
import datetime

print(datetime.datetime.now())
print(tf.__version__)
print(tf.config.list_physical_devices())
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
print(datetime.datetime.now())