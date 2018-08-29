import random
import numpy as np
import tensorflow as tf

from data import *
from model import SegCaps
from config import get_config
from utils import ImageMaskGenerator
#import tensorflow.contrib.eager as tfe

#TODO: update code for Eager Execution

#tf.enable_eager_execution()
#print("TensorFlow version: {}".format(tf.VERSION))
#print("Eager execution: {}".format(tf.executing_eagerly()))

#adpapted for Siamese NN for One-Shot Image Recognition:
def contrastive_loss(model1, model2, y, margin):
	with tf.name_scope("contrastive-loss"):
		d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=True))
		tmp= y * tf.square(d)
		tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
		return tf.reduce_mean(tmp + tmp2) /2

def triplet_loss(model1):
  return None

#Hyperparameters:
#train_loss_results=[]
#train_accuracy_results=[]

#keep results for plotting
#train_loss_results=[]
#train_accuracy_results=[]
num_epochs=201

def main():
  # Get configuration
  config = get_config()
  tf_config = tf.ConfigProto(allow_soft_placement=True)
  tf_config.gpu_options.visible_device_list = config.device
  #with tf.Session(init) as sess:
  with tf.Session(config=tf_config) as sess:
    #for d in ['/device:GPU:0','/device:GPU:1','/device:GPU:2']:
    with tf.device('/device:GPU:0'):
      # model = SegCaps(sess, config, is_train=True)
      # with tf.variable_scope("model") as scope:
      model = SegCaps(sess, config, is_train=True)
      sess.run(tf.global_variables_initializer())
      # with tf.variable_scope(scope, reuse=True):
      # model2 = SegCaps(sess, config, is_train=True)
      images, labels = ImageMaskGenerator('dataset/annotations/val2017/', 'dataset/images/val2017/')
      print(images.shape)
      print(labels.shape)
      for epoch in range(num_epochs):
        loss_value = model.fit(images, labels)
        print(loss_value)
        model.save()

if __name__=="__main__":
  # To reproduce the result
  tf.set_random_seed(2018)
  np.random.seed(2018)
  random.seed(2018)
  main()
