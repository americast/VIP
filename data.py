# -*- coding: utf-8 -*-
# @Author: Saurabh Agarwal
# @Date:   2018-05-24 10:39:32
# @Last Modified by:   Saurabh Agarwal
# @Last Modified time: 2018-05-28 02:55:14
from skimage import io
import skimage.transform
import os
import glob
import numpy as np

BATCH_SIZE = 10

def data_in_batches():
	print("Boat is stud!")
	XPath = os.path.join(".","data/train/train")
	YPath = os.path.join(".","data/train/GT")

	XFileName = glob.glob(os.path.join(XPath,"*.tiff"))
	YFileName = glob.glob(os.path.join(YPath,"*.tiff"))

	print("Yo boat!"+str(XFileName))
	x = []
	y = []
	print(len(XFileName))
	for file in XFileName[:1]:
		# print file
		img = io.imread(file)
		resized = skimage.transform.resize(img, (64, 640, 960))
		x.extend(resized)
		# print len(x)
		# print shape(x)
	x = np.asarray(x)
	X = np.reshape(x, [1, x.shape[0],x.shape[1], x.shape[2]])

	# X = np.append(x,x,axis=3)
	# X = np.append(X,x,axis=3)
	print(X.shape)

	for file in YFileName:
		# print file
		img = io.imread(file)
		resized = skimage.transform.resize(img, (64, 640, 960))
		y.extend(resized)
		# print len(x)
		# print shape(x)
	y = np.asarray(y)
	y = np.reshape(y, [1, y.shape[0],y.shape[1], y.shape[2]])
	print(y.shape)

	indices = np.random.permutation(X.shape[0])
	# print(indices)

	X = X[indices]
	y = y[indices]

	# count = 0
	# for i in xrange(X.shape[0]/BATCH_SIZE):
	# 	yield (X[count:count+BATCH_SIZE], y[count:count+BATCH_SIZE])
	# 	count+=BATCH_SIZE

# print("Boat is not stud!")
if __name__ == '__main__':
	data_in_batches()
	# for i in data_in_batches():
	  # print(i)