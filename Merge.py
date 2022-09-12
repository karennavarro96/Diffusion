import sys
import glob
import pandas as pd

filewildcard = sys.argv[1]
files = glob.glob(filewildcard)

DF = []

for f in files:
    DF.append(pd.read_hdf(f,"/CHITS/lowTh"))
    
DF_m = pd.concat(DF)

DF_m.to_hdf("merge.h5",'/CHITS/lowTh')