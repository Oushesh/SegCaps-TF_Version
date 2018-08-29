#read in images
#read in Mask
from PIL import Image
import os

base_path='../Data/valid/val2017/'

#Oushesh Haradhun
#the pixel value of the segmentation pixel is the class ID of MS COCO Dataset
#Read the annotation images
#Convert to one hot encoding vector
import os
import numpy as np
from PIL import Image
#one hot encoding from Keras: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

batch_size=2
num_classes=256
def mask2onehot(path):
	maskbatch=[]
	maskbatch=np.zeros((batch_size,512,512,num_classes))
	for files in os.listdir(path):
		i=0
		print (files)
		im=Image.open(path + files)
		#reshape to 512,512
		im=im.resize((512,512),Image.ANTIALIAS)
		im= np.asarray(im)
		#read the pixel and it is the classID
		#convert image Labels(classID) to one-hot encoding
		encoded = to_categorical(im, num_classes=256)
		print ("one hot encoded:",encoded)
		print ("image encoded shape:",encoded.shape)
		maskbatch[i]=encoded
		i+=1
	print ("maskbatch:", maskbatch.shape)
	return maskbatch
#input=mask2onehot('../dataset/annotations/val2017/')

"""Encoded is the image one hot encoding for the images"""
def generatorfromDirectory():
	"""Dataset in the Format: Data-> train/class_a, train/class_b, train/class_c, etc.. """
	seed = 1
	data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
	train_image_generator = image_datagen.flow_from_directory(
	    '../dataset/images/train/train2017',
	    class_mode=None,
	    seed=seed)

	train_mask_generator = mask_datagen.flow_from_directory(
	    '../dataset/annotations/train2017',
	    class_mode=None,
	    seed=seed)

	val_image_generator = image_datagen.flow_from_directory(
	    '../dataset/images/val2017',
	    class_mode=None,
	    seed=seed)

	val_mask_generator = mask_datagen.flow_from_directory(
	    '../dataset/annotations/val2017',
	    class_mode=None,
	    seed=seed)
	return train_image_generator, train_mask_generator

#generatorfromDirectory()
def generatorMSCOCOmask(input):
	datagen = ImageDataGenerator(
	    featurewise_center=True,
	    featurewise_std_normalization=True,
	    rotation_range=20,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    horizontal_flip=True)

	batches=datagen.fit(input)
	print ("successfull")
	return batches

#generatorMSCOCOmask(mask2onehot('../dataset/annotations/val2017/'))

def generatorImage(path):
	Imagebatch=np.zeros((batch_size,512,512,3))

	for files in os.listdir(path):
		i = 0
		print (files)
		im=Image.open(path + files)
		# reshape to 512,512
		im = im.resize((512, 512), Image.ANTIALIAS)
		if len(np.asarray(im)!=3):
			print ("Grey Image Found")
			im=np.resize(im,(np.asarray(im).shape[0],np.asarray(im).shape[1],3))
		im= np.asarray(im)
		print ("image encoded shape:",im.shape)
		Imagebatch[i]=im
		i+=1
	print ("Imagebatch:", Imagebatch.shape)
	return Imagebatch

def ImageMaskGenerator(maskpath,imagepath):
	Imagebatch=generatorImage(imagepath)
	maskbatch=mask2onehot(maskpath)
	print ("maskbatch shape:",maskbatch.shape)
	print ("imagebatch shape:", Imagebatch.shape)
	#yield Imagebatch,maskbatch)
	return Imagebatch, maskbatch

#ImageMaskGenerator('dataset/annotations/val2017/','dataset/images/val2017/')


