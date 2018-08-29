import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import backend as K
from scipy.spatial.distance import directed_hausdorff

# def d_h_av(y_true, y_pred):
# 	for each in y_true:
# 		dist = fabs(y_true[0] - y_pred[0])
# 		for every in y_pred:
# 			dist_here = fabs(each - every)
# 			if (dist_here < dist):
# 				dist = dist_here



	# u = np.asarray(y_true, dtype=np.float64, order='c')
	# return (directed_hausdorff(y_true, y_pred)[0] + directed_hausdorff(y_pred, y_true)[0]) / 2


# def dice_coef(y_true, y_pred, smooth=0.0):
#     '''Average dice coefficient per batch.'''
#     axes = (1,2,3)
#     intersection = K.sum(y_true * y_pred, axis=axes)
#     summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
#     return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

# def ninety_five_h(y_true, y_pred):
# 	to_sum = []
# 	sum_ = 0
# 	for each in y_true:
# 		to_sum.append(directed_hausdorff(each, y_pred)[0])

# 	to_sum.sort()
# 	to_sum = to_sum[:int(len(to_sum)*0.75)]
# 	sum_ = 0.5 * sum(to_sum)

# 	to_sum = []
# 	sum_ = 0
# 	for each in y_pred:
# 		to_sum.append(directed_hausdorff(each, y_true)[0])

# 	to_sum.sort()
# 	to_sum = to_sum[:int(len(to_sum)*0.75)]
# 	sum_ += 0.5*sum(to_sum)

# 	return sum_


def switch_mean_iou(labels, predictions):
    """
    labels,prediction with shape of [batch,height,width,class_number=2]
    """
    mean_iou = K.variable(0.0)
    seen_classes = K.variable(0.0)

    for c in range(2):
        labels_c = K.cast(K.equal(labels, c), K.floatx())
        pred_c = K.cast(K.equal(predictions, c), K.floatx())

        labels_c_sum = K.sum(labels_c)
        pred_c_sum = K.sum(pred_c)

        intersect = K.sum(labels_c*pred_c)
        union = labels_c_sum + pred_c_sum - intersect
        iou = intersect / union
        condition = K.equal(union, 0)
        mean_iou = K.switch(condition,
                            mean_iou,
                            mean_iou+iou)
        seen_classes = K.switch(condition,
                                seen_classes,
                                seen_classes+1)

    mean_iou = K.switch(K.equal(seen_classes, 0),
                        mean_iou,
                        mean_iou/seen_classes)
    return mean_iou

def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    # merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = Concatenate(axis = 3)([drop4, up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = Concatenate(axis = 3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis = 3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis = 3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [switch_mean_iou])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


