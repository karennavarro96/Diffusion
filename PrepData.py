import pandas as pd
import h5py
import numpy as np
import os
import json
import sys
import glob

para = {
    "train_test_split": 0.8,
    "epochs": 30,
    "high_energy_thres": 1.5,
    "radius_thres": 217,
    "z_thres_lower": 20,
    "z_thres_upper": 644
}

data = pd.read_hdf("merge.h5", "/CHITS/lowTh")

# Drop uneeded columns
data = data.drop(columns=['time', 'npeak', 'Xpeak', 'Ypeak', 'nsipm', 'Xrms', 'Yrms', 'Q', 'E', 'Qc', 'track_id', 'Ep'])

# Add radius to table
data['r'] = np.sqrt(data['X']**2 + data['Y']**2)

# Replace nan to 0
data['Ec'] = data['Ec'].fillna(0)

# Apply cuts
dataZlow = data[(data.Z <= para["z_thres_lower"])]['event'].unique()
dataZhigh = data[data.Z >= para["z_thres_upper"]]['event'].unique()
datar = data[(data.r >= para["radius_thres"])]['event'].unique()

data['Esum'] = data.groupby(["event"])["Ec"].transform('sum')
dataE = data[(data.Esum < para["high_energy_thres"])]['event'].unique()

# Filter the dataframe based on cut results
data = data[~data.event.isin(dataZlow)]
data = data[~data.event.isin(dataZhigh)]
data = data[~data.event.isin(datar)]
data = data[~data.event.isin(dataE)]

# Drop uneeded columns
data = data.drop(columns=['r', "Esum"])


passed = []

# Apply DL image cut
for event in data.event.unique():
    my_event = data[data.event == event]

    image_3d = np.stack((my_event.X, my_event.Y, my_event.Z), axis=1)
    db = DBSCAN(eps=20, min_samples=30).fit(image_3d)
    labels = db.labels_

    # if the event only has a single cluster and no background OR
    # if the event has two clusters and one of those is just the background
    # put the event in the single track file, else, move on to next event
    if len(np.unique(labels)) == 1 and np.unique(labels)[0] != -1: 
        pass
    elif len(np.unique(labels)) == 2 and -1 in np.unique(labels):
        pass
    else:
        passed.append(event)
        continue

data = data[~data.event.isin(passed)]

print(len(data.event.unique())) 

data.to_hdf( "./strack.h5", "/CHITS/lowTh", format='t', append=True, data_columns=True)