import cv2

import os
import random
import skimage.io
from PIL import Image
import numpy as np
from scipy import ndimage
from skimage.transform import resize
class MSCOCO(object):
  def __init__(self, config, is_train):
    self.train_volume_path = os.path.join(config.data_dir, 'train/imgs/')
    self.train_labels_path = os.path.join(config.data_dir, 'train/masks/')
    self.test_volume_path = os.path.join(config.data_dir, 'test')
    self.batch_size = config.batch_size

    if is_train:
      self.volume = self.preprocessing(skimage.io.imread(self.train_volume_path + 'train1.png'))
      self.labels = self.preprocessing(skimage.io.imread(self.train_labels_path + 'train1.png'))
      if config.mask_inv:
        self.labels = 1. - self.labels

      self.idx_list = range(self.volume.shape[0])
    else :
      self.volume = self.preprocessing(skimage.io.imread(self.test_volume_path))
      return self.volume, self.labels

  def random_sample(self):
    idx = random.sample(self.idx_list, self.batch_size)
    return self.volume[idx],self.labels[idx]
    #return self.data_aug(self.volume[idx], self.labels[idx])

  def get_volume(self, idx):
    return self.volume[idx:idx+1]
  
  @staticmethod
  def preprocessing(x):
    x=resize(x,(512,512),anti_aliasing=True)
    return x
    #return np.expand_dims(x.astype(np.float32) / 255., axis=3)

  def data_aug(self, x, y):
    x_aug = np.empty_like(x)
    y_aug = np.empty_like(y)
    for i in range(x.shape[0]):
      x_ = x[i,:,:,0] 
      y_ = y[i,:,:,0]

      if np.random.randint(2):
        x_ = x_[::-1,:]
        y_ = y_[::-1,:]

      # rotation
      rot_num = np.random.randint(4)
      x_ = np.rot90(x_, rot_num)
      y_ = np.rot90(y_, rot_num)

      # elastic deformation
      size = 8
      ampl = 8
      du = np.zeros([size, size])
      dv = np.zeros([size, size])
      du[1:-1,1:-1] = np.random.uniform(-ampl, ampl, size=(size-2, size-2))
      dv[1:-1,1:-1] = np.random.uniform(-ampl, ampl, size=(size-2, size-2))

      DU = cv2.resize(du, (self.volume.shape[1], self.volume.shape[2]))
      DV = cv2.resize(du, (self.volume.shape[1], self.volume.shape[2]))
      U, V = np.meshgrid(
        np.arange(self.volume.shape[1]),
        np.arange(self.volume.shape[2]))
      indices = np.reshape(V+DV, (-1, 1)), np.reshape(U+DU, (-1, 1))

      x_ = ndimage.interpolation.map_coordinates(x_, indices, order=1).reshape(x_.shape)
      y_ = ndimage.interpolation.map_coordinates(y_, indices, order=1).reshape(y_.shape)
      print (np.transpose(x_).shape)

      # Gaussian noise + Gaussian Blur
      #x_ = x_ + np.random.normal(0., 0.1)
      #x_ = cv2.GaussianBlur(x_, (0, 0), np.random.uniform(0., 2.))
      x_transpose=np.transpose(x_)
      x_aug[i,:,:,0] = x_transpose
      y_aug[i,:,:,0] = np.transpose(y_)

    return  x_aug, y_aug

  def ImageRead(path):
      im=Image.open(path+'imgs/train1.png')
      #reshape to 512,512
      im=im.resize((512,512),Image.ANTIALIAS)
      im= np.asarray(im)
      mask=Image.open(path+'masks/train1.png')
      mask =np.asarray(mask)
      mask[mask>=1]=1
      mask[mask!=0]=0
      print ("maskbatch:", maskbatch.shape)
      return im,mask


#MSCOCO.ImageRead('dataset/train/')