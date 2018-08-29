from model import *
from data import *
import numpy as np
import pudb
x = np.load("X_val.npy")
Y = np.load("y_val.npy")


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')

# data_gen_args = dict()
# myGene = trainGenerator(2,'data/train/','train','Labels',data_gen_args,save_to_dir = None)
# print "Hello"
# model = unet()
# print "Hello2"
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# print "Hello3"

# checkpointer = ModelCheckpoint(monitor='val_loss', filepath=model_file, verbose=True,
                               # save_best_only = True)
# earlystopping = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=20, verbose=True)
# model.fit(X_train, X_train, epochs=200, batch_size=100, shuffle=False,
                    # validation_data=(X_val, X_val), verbose=True, callbacks=[checkpointer, 
# earlystopping])
# model.fit(x,Y,batch_size=4,epochs=1,verbose=1, callbacks=[model_checkpoint])
# model.fit_generator(myGene,steps_per_epoch=1,epochs=1,callbacks=[model_checkpoint])

model = load_model("unet_membrane.hdf5_copy",custom_objects={"switch_mean_iou": switch_mean_iou})
print "Hello4"


# testGene = testGenerator("data/train/test")
# print "Hello5"
#results = model.predict_generator(testGene,30,verbose=1)


predicted = model.predict(x,verbose=True, batch_size=1)
np.save("testing.npy",predicted)


print(predicted.shape)


score = model.evaluate(x, Y, batch_size=4)
print(model.metrics_names)
print("score: "+str(score))




# result = model.predict(x,batch_size=1,verbose=1)
# np.save('result.npy',result)
# print "Hello6"
saveResult("data_val/train/test",predicted)
# print "Hello7"
# pu.db
