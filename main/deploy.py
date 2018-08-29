import os
import numpy as np
import tensorflow as tf
import skimage.io

from data import *
from model import SegCaps
from config import get_config

def main():
  # Get configuration
  config = get_config()
  config.batch_size = 1

  # Data reader
  if config.dataset=="isbi2012":
    data_reader = ISBI2012Reader(config, is_train=False)
  else:
    raise ValueError("No such dataset.")

  tf_config = tf.ConfigProto(allow_soft_placement=True)
  tf_config.gpu_options.visible_device_list = config.device
  with tf.Session(config=tf_config) as sess:
    model = SegCaps(sess, config, is_train=False)
    sess.run(tf.global_variables_initializer())
    model.restore()

    if not tf.gfile.Exists(config.result_dir):
      tf.gfile.MakeDirs(config.result_dir)

    for i in range(data_reader.volume.shape[0]):
      image = data_reader.get_volume(i)
      label = model.predict(image)[0,:,:,0]
      if config.mask_inv:
        label = 1. - label
      result_path = os.path.join(config.result_dir, "{:02d}.tif".format(i+1))
      skimage.io.imsave(result_path, (label * 255.).astype(np.uint8))

if __name__=="__main__":
  main()
