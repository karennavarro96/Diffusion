import pandas as pd
import numpy as np
import gc
from random import shuffle
import math
import json
import h5py
import inspect
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits import mplot3d
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import pickle
warnings.filterwarnings('ignore')

cur_dir = './'

x_range, y_range, z_range = 270.0, 280.0, 255
X_shape = int(x_range/10) +1
Y_shape = int(y_range/10) 
print(X_shape, Y_shape, x_range, y_range, z_range)

num = X_shape * Y_shape * (z_range+1)       #train[0].shape[0]=
print(num)

para = {
    "train_test_split": 0.8,
    "epochs": 30,
    "high_energy_thres": 1.5,
    "radius_thres": 217,
    "z_thres_lower": 20,
    "z_thres_upper": 644
}

val_split = 1 - para['train_test_split']
num_epochs = para['epochs']
# Function to resize image to 64x64
row, col, ch = X_shape, Y_shape, int(num/(X_shape*Y_shape))
model = Sequential()
model.add(tensorflow.keras.layers.Reshape((row, col, ch,), input_shape=(row*col*ch,)))
model.add(tensorflow.keras.layers.Permute((2,3,1), input_shape=(row, col,ch)))

model.add(tensorflow.keras.layers.Convolution2D(filters= 4, kernel_size =(4,4), strides= (1,1), padding='same', name='conv1')) #96
model.add(tensorflow.keras.layers.Activation('relu'))
model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1), name='pool1'))


model.add(Flatten())
model.add(tensorflow.keras.layers.Dropout(0.5))

model.add(Dense(32, name='dense2'))  #1024
model.add(tensorflow.keras.layers.Activation('relu'))

model.add(Dense(1,name='output'))
model.add(tensorflow.keras.layers.Activation('relu'))


model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0001), loss='mean_squared_error')

model.summary()

#Data Ec COG
epochs=30
passes=3
Files=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11] #9
histories=[]
for p in range(0,passes):
  print("Pass "+str(p))
  for i in Files:
    infile= open(cur_dir+"/traindatanorm_z_"+str(i)+".pkl",'rb')
    train, output=pickle.load(infile)
    #note output[:,0] is centroid and output[:,1] is COG
    histories.append(model.fit(train, output[:,1],  validation_split=0.0, epochs=epochs, shuffle=True))
    model.fit(train, output[:,1],  validation_split=0.0, epochs=epochs, shuffle=True)
    infile.close()
    gc.enable() # enable manual garbage collection
    gc.collect() # check for garbage collection

model.save_weights(cur_dir+ "ModelYZ.h5")
model_json = model.to_json()
with open(cur_dir+ "ModelYZ.json", "w") as json_file:
    json_file.write(model_json)

# Plot training & validation loss values
for i in range(0,len(histories)):
  history=histories[i]
  plt.plot(np.arange(0,len(history.history['loss']))+i*epochs, history.history['loss'],color='red')
  #plt.plot(np.arange(0,len(history.history['loss']))+i*epochs, history.history['val_loss'],color='blue')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
