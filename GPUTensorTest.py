CUDA_VISIBLE_DEVICES=0
import tensorflow as tf
print("TensorFlow Version:", tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())