import pandas as pd
import numpy as np
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
data = pd.read_hdf(cur_dir + 'strack.h5', '/CHITS/lowTh')
data=data[(data.X!=0) * (data.Y!=0)]
data=data[~np.isnan(data.Ec)]
data=data[~np.isnan(data.X)]

x_range, y_range, z_range = 270.0, 280.0, 255
X_shape = int(x_range/10) +1
Y_shape = int(y_range/10) 
print(X_shape, Y_shape, x_range, y_range, z_range)

# The original data has coordinates in multiples of 10 in X & Y. When you take that and convert
# to a numypy array to make an "image array", it takes a lot of space, the code below, hence,
# divides by that to make the output image smaller
def coord_to_image(X, Y, E):
    image = np.zeros((X_shape, Y_shape))
    for i in range(len(X)):
        if (X[i]/10 < X_shape ) and (Y[i]/10 < Y_shape):
            if (X[i]%10 == 0) and (Y[i]%10 == 0):
                image[int(Y[i]/10)][int(X[i]/10) ] = E[i]
        #else: (X[i]%10 == 280) and (Y[i]%10 == 280):
        #   image[int(Y[i]/10)][int(X[i]/10)] = Ec[i]
    return image

# After the transformations are applied, the size of the image is already cropped from before, so
# we don't need that division by 10 here
def coord_to_image_plain(X, Y, E):
    image = np.zeros((X_shape, Y_shape))
    for i in range(len(X)):
        image[int(Y[i])][int(X[i])] = E[i]
    return image

def image_to_coord(image):
    Z = np.nonzero(image)[0]
    Y = np.nonzero(image)[1]
    X = np.nonzero(image)[2]
    E = image[np.where(image!=0)]
    return X, Y, Z, E

# This class is used to make various transformations of the original event to get more samples to train on
# The keras' imagedatagenerator method sometimes repeats the original image, which may have lead to 
# over-training so I thought it would be best to code the transformations myself
# The doc line for the method flip_z that says "flip_z" should not be changes as process_data uses
# that information to change the value that the model should predict for that particular
# transformation
class Transform(object):
    def __init__(self, image):
        """init"""
        self.image = image
        return None
    def no_change(self):
        X, Y, Z, E = image_to_coord(self.image)
        Z = Z - min(Z)
        return X, Y, Z, E
    def flip_y(self):
        y_image = np.flip(self.image, 1)
        X, Y, Z, E = image_to_coord(y_image)
        Y = Y - min(Y)
        return X, Y, Z, E
    def flip_x(self):
        x_image = np.flip(self.image, 2)
        X, Y, Z, E = image_to_coord(x_image)
        X = X - min(X)
        return X, Y, Z, E
    def flip_xy(self):
        x_image = np.flip(np.flip(self.image, 2),1)
        X, Y, Z, E = image_to_coord(x_image)
        X = X - min(X)
        return X, Y, Z, E
    def rotate90(self):
        x_image = np.swapaxes(self.image, 1,2)
        X, Y, Z, E = image_to_coord(x_image)
        X = X - min(X)
        return X, Y, Z, E
    def rotate90_flip_x(self):
        x_image = np.swapaxes(np.flip(self.image, 2), 1,2)
        X, Y, Z, E = image_to_coord(x_image)
        X = X - min(X)
        return X, Y, Z, E
    def rotate90_flip_y(self):
        x_image = np.swapaxes(np.flip(self.image, 1), 1,2)
        X, Y, Z, E = image_to_coord(x_image)
        X = X - min(X)
        return X, Y, Z, E
    def rotate90_flip_xy(self):
        x_image = np.swapaxes(np.flip(np.flip(self.image, 2),1), 1,2)
        X, Y, Z, E = image_to_coord(x_image)
        X = X - min(X)
        return X, Y, Z, E

# This function takes in X,Y,Z,E coordinates and converts that to a numpy array
# It slices the Z coordinates based on a range, so for instance, if your Z coordinates are
# 1.2, 1.4, 2.5, 2.6, 2.7
# and your slice happens to be of "size 1", the coordinates will conver to:
# 1, 1, 2, 2, 2
# z_step determines the width of the slices, the smallest this argument can be is 1
def convert_to_3D(X_, Y_, Z_, E_, z_step):
    my_event = pd.DataFrame({'X' : X_, 'Y' : Y_, 'Z' : Z_, 'Ec' : E_})
    Z = my_event.Z
    event_image = []
    for z in range(math.floor(min(Z)), math.floor(min(Z)) + z_range + z_step, z_step):
        z_slice = Z[(Z >= z) & (Z < z+z_step)]
        z_slice = z_slice.unique()
        num_z_slices = len(z_slice)
        if num_z_slices == 0:
            X = [0, 0]
            Y = [0, 0]
            E = [0, 0]
            image_x_y = coord_to_image(X, Y, E)
        else:
            z_count = 0
            for each_z in z_slice:
                event_slice = my_event[my_event.Z == each_z]
                X = list(event_slice.X)
                Y = list(event_slice.Y)
                E = list(event_slice.Ec)
                if z_count == 0:
                    image_x_y = coord_to_image(X, Y, E)
                    z_count += 1
                else:
                    image_x_y += coord_to_image(X, Y, E)
        event_image.append(image_x_y)
    return np.array(event_image), image_x_y.shape[0], image_x_y.shape[1]

######

######

# This is only slightly different from above since we do not need to crate z-slices anymore
# The z-coordinates (when this function is called) are already integer slices, so this doesn't
# bother with the slicing
def convert_to_3D_plain(X_, Y_, Z_, E_, N):
    my_event = pd.DataFrame({'X' : X_, 'Y' : Y_, 'Z' : Z_, 'Ec' : E_})
    event_image = []
    for z in range(math.floor(min(Z_)), max(Z_)):
        X = list(my_event[my_event.Z == z].X)
        Y = list(my_event[my_event.Z == z].Y)
        E = list(my_event[my_event.Z == z].Ec)
        image_x_y = coord_to_image_plain(X, Y, E)
        event_image.append(image_x_y)
    for z in range(N - (max(Z_) - min(Z_))):
        X = [0, 0]
        Y = [0, 0]
        E = [0, 0]
        image_x_y = coord_to_image(X, Y, E)
        event_image.append(image_x_y)
    return np.array(event_image)

def process_data_validation(data, z_step, counter_start=0, counter_end=len(data.event.unique()),FixE=537413):
    train_data = [] 
    train_output = []
    counter = counter_start
    for event in data.event.unique()[counter_start:counter_end]:
        print("Onto #" + str(counter))
        my_event = data[data.event == event]
        # this part takes the single track that was identified in prep_data and excludes
        # all the background to get only the single track to be trained by the network
        image_3d = np.stack((my_event.X, my_event.Y, my_event.Z), axis=1)
        db = DBSCAN(eps=20, min_samples=30).fit(image_3d)
        clusters = np.array(db.fit_predict(image_3d))
        indices = np.array([i for i, x in enumerate(clusters) if x == 0])
        # This part centers the entire event to ensure that the information about the 
        # true z coordinates (with the S1 signal) is removed from the event before it is
        # fed to the network
        X_ = np.array(my_event.X)[indices] - min(np.array(my_event.X)[indices])
        Y_ = np.array(my_event.Y)[indices] - min(np.array(my_event.Y)[indices])
        Z_ = np.array(my_event.Z)[indices] - min(np.array(my_event.Z)[indices])
        E_ = np.array(my_event.Ec)[indices]
        if(FixE>0):
          E_=E_/np.nansum(E_)*FixE
        event_image, im_x_y_x, im_x_y_y = convert_to_3D(X_, Y_, Z_, E_,z_step)
        T = Transform(event_image)
        attrs = (getattr(T, name) for name in dir(T))
        methods = filter(inspect.ismethod, attrs)
        for method in methods:
            try:
                if method.__doc__ != "init":
                   # try:
                    X, Y, Z, E = method()
                   # except:
                        # there are some events that give an error, I haven't looked into them, but 
                        # there's only a handful so I chose to ignore them for the time being
                        #fig = plt.figure()
                        #ax = fig.add_subplot(111, projection='3d')
                        #ax.scatter(X, Y, Z)
                    test_image = convert_to_3D_plain(X, Y, Z, E, event_image.shape[0])
                    test_image = np.reshape(test_image, -1)
                    train_data.append(test_image)
                    # this line ensures that the Z coordinate the network is trained on is of the uncropped event
                    # so it gets the true Z position
                    my_event = data[data.event == event]
                    #COG=sum(my_event.Z*my_event.E)/sum(my_event.E)
                    COG=np.nansum(my_event.Z*my_event.Ec)/np.nansum(my_event.Ec)
                    centroid = my_event.Z.mean()
                    #E=sum(my_event.E)
                    Etotal=np.nansum(my_event.Ec)
                    Lx=max(my_event.X)-min(my_event.X)
                    Ly=max(my_event.Y)-min(my_event.Y)
                    Lz=max(my_event.Z)-min(my_event.Z)
                    Ox=max(my_event.Z)-centroid
                    R=(my_event.X.mean()**2+my_event.Y.mean()**2)**0.5
                    train_output.append([centroid,COG,Etotal,Lx,Ly,Lz,R,Ox])
                    #print(len(train_output))
                    #train_output.append(sum(E))
                    counter += 1
            except TypeError:
            # cannot handle methods that require arguments
                pass
    print(np.sum(np.isnan(train_data)))
    return np.array(train_data), np.array(train_output)

# for reasons that are unclear, I could not get more than around 3600 events on a
# single variable. It gives me a memory error but clearly the memory is not getting
# full if you look to top right, but having the data split between two variables
# works. Note that this is a separate issue from training on large data sets
# This is just about being able to load data into memory, the other issue is training
# such a large dataset on the GPU
AveE=537413
num_samples = 3600 #3600
SampleChunk= 300 #300
#for i in range(0, 12): # normally, 0 to int(num_samples/SampleChunk)
for i in range(0, int(num_samples/SampleChunk)): 
  train_data, train_output = process_data_validation(data, 1, i*SampleChunk, (i+1)*SampleChunk, FixE=537413)
  outf=open(cur_dir+"/traindatanorm_z_"+str(i)+".pkl",'wb')
  print("Wrote "+ cur_dir+"/traindatanorm_z_"+str(i)+".pkl")
  pickle.dump([train_data, train_output],outf, protocol=4)
  outf.close()

def plot_them(data, z_step, counter_start=0, counter_end=len(data.event.unique()),FixE=537413):
    train_data = []
    train_output = []
    counter = counter_start
    for event in data.event.unique()[counter_start:counter_end]:
        print("Onto #" + str(counter))
        my_event = data[data.event == event]
        # this part takes the single track that was identified in prep_data and excludes
        # all the background to get only the single track to be trained by the network
        image_3d = np.stack((my_event.X, my_event.Y, my_event.Z), axis=1)
        db = DBSCAN(eps=20, min_samples=30).fit(image_3d)
        clusters = np.array(db.fit_predict(image_3d))
        indices = np.array([i for i, x in enumerate(clusters) if x == 0])
        # This part centers the entire event to ensure that the information about the 
        # true z coordinates (with the S1 signal) is removed from the event before it is
        # fed to the network
        fig, axs = plt.subplots(2, 4, sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': 0},figsize=(8,5),dpi=150)
        
        axsf= axs.flatten()   
        X_ = np.array(my_event.X)[indices] - min(np.array(my_event.X)[indices])
        Y_ = np.array(my_event.Y)[indices] - min(np.array(my_event.Y)[indices])
        Z_ = np.array(my_event.Z)[indices] - min(np.array(my_event.Z)[indices])
        E_ = np.array(my_event.Ec)[indices]
        if(FixE>0):
          E_=E_/sum(E_)*FixE
        event_image, im_x_y_x, im_x_y_y = convert_to_3D(X_, Y_, Z_, E_,z_step)
        T = Transform(event_image)
        attrs = (getattr(T, name) for name in dir(T))
        methods = filter(inspect.ismethod, attrs)
        axi=0
        for method in methods:
            try:
                if method.__doc__ != "init":
                   # try:
                    X, Y, Z, E = method()
                    Eth= 1.5
                    print(method.__doc__)
                    axsf[axi].scatter(X[E>Eth]-np.mean(X[E>Eth]),Y[E>Eth]-np.mean(Y[E>Eth]),c=E[E>Eth],vmin=250, vmax=1000,s=10,label=method.__name__)
                    axsf[axi].set_title(method.__name__)

                    axi+=1

                   # except:
                        # there are some events that give an error, I haven't looked into them, but 
                        # there's only a handful so I chose to ignore them for the time being
                        #fig = plt.figure()
                        #ax = fig.add_subplot(111, projection='3d')
                        #ax.scatter(X, Y, Z)
                    test_image = convert_to_3D_plain(X, Y, Z, E, event_image.shape[0])
                    test_image = np.reshape(test_image, -1)
                    train_data.append(test_image)
                    # this line ensures that the Z coordinate the network is trained on is of the uncropped event
                    # so it gets the true Z position
                    my_event = data[data.event == event]
                    COG=np.nansum(my_event.Z*my_event.Ec)/np.nansum(my_event.Ec)
                    a = np.nansum(my_event.Ec)
                    centroid = my_event.Z.mean()
                    maxZ=max(my_event.Z())
                    train_output.append([centroid,COG,maxZ])
                    #train_output.append(sum(E))
                    counter += 1
            except TypeError:
            # cannot handle methods that require arguments
                pass
        #axsf[0].colorbar()
        plt.title(a)
        plt.show()
    return np.array(train_data), np.array(train_output)

train_data, train_output = plot_them(data, 4, 5, 10)

train_data[0].shape[0]

x = np.array(train_data)
y = train_output

for i in x:
    print(i)

print(len(train_data))
len(train_output)