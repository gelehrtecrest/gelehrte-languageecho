from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import glob
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)