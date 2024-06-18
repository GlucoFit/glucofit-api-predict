import tensorflow as tf
from models.model1 import load_model1
from models.model2 import load_model2
# from models.model3 import load_model3

# Set TensorFlow logging verbosity
tf.get_logger().setLevel('INFO')

print('loading model 1')
model1 = load_model1() # Deep Learning : CBF MRS

print('loading model 2')
model2 = load_model2() # ResNet 50 : IR

print('success')
# model3 = load_model3()
