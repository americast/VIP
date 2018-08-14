from model import *
from data import *
import numpy as np
x = np.load("X.npy")
Y = np.load("y.npy")
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')

data_gen_args = dict()
myGene = trainGenerator(2,'data/train/','train','Labels',data_gen_args,save_to_dir = None)
print "Hello"
model = unet()
print "Hello2"
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
print "Hello3"
model.fit(x,Y,batch_size=1,epochs=1)
# model.fit_generator(myGene,steps_per_epoch=1,epochs=3,callbacks=[model_checkpoint])
print "Hello4"

testGene = testGenerator("data/train/test")
print "Hello5"
results = model.predict_generator(testGene,30,verbose=1)
print "Hello6"
saveResult("data/train/test",results)
print "Hello7"