import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU is ready to be used: {gpus[0].name}")
else:
    print("GPU is NOT available.")

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

import tensorflow as tf
from tensorflow.python.client import device_lib

print("Available devices:")
print(device_lib.list_local_devices())