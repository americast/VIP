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
	XPath = os.path.join(".","data/train/train")
	YPath = os.path.join(".","data/train/GT")

	XFileName = glob.glob(os.path.join(XPath,"*.tiff"))
	YFileName = glob.glob(os.path.join(YPath,"*.tiff"))

	x = []
	y = []

	for file in XFileName:
		img = io.imread(file)
		resized = skimage.transform.resize(img, (64, 640, 960))
		x.insert(0,resized)
	X= np.asarray(x)

	for file in YFileName:
		img = io.imread(file)
		resized = skimage.transform.resize(img, (64, 640, 960))
		y.insert(0,resized)
	y = np.asarray(y)
	np.save('X.npy',X)
	np.save('y.npy',y)

if __name__ == '__main__':
	data_in_batches()
