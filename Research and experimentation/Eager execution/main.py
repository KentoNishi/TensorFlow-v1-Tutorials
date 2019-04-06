from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import tempfile

# Eager execution is enabled by default on TF2.0
# tf.enable_eager_execution()

# numpy compatibility
ndarray = np.ones([3, 3])
print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)
print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))
print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

# GPU acceleration
x = tf.random.uniform([3, 3])
print("Is there a GPU available: "),
print(tf.test.is_gpu_available())
print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

# Device names
def time_matmul(x):
	start = time.time()
	for loop in range(10):
		tf.matmul(x, x)
	result = time.time()-start
	print("10 loops: {:0.2f}ms".format(1000*result))
# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
	x = tf.random.uniform([1000, 1000])
	assert x.device.endswith("CPU:0")
	time_matmul(x)
# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
	with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
		x = tf.random.uniform([1000, 1000])
		assert x.device.endswith("GPU:0")
		time_matmul(x)

# Create a CSV file
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
_, filename = tempfile.mkstemp()
with open(filename, 'w') as f:
	f.write("""Line 1
	Line 2
	Line 3
	""")
ds_file = tf.data.TextLineDataset(filename)
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)
print('Elements of ds_tensors:')
for x in ds_tensors:
	print(x)
print('\nElements in ds_file:')
for x in ds_file:
	print(x)