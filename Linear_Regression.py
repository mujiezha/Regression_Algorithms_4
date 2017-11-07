#there are 4 algorithm :Perceptron Learning algorithm,  Pocket algorithm , Logistic Regression, Linear Regression
#  Implement the Perceptron Learning algorithm. Run it on the data file "classification.txt" ignoring the 5th column.
# That is, consider only the first 4 columns in each row.
# The first 3 columns are the coordinates of a point; and the 4th column is its classification label +1 or -1.
# Report your results.

import matplotlib.pyplot as plt
import random
import numpy as np
import math
import pandas as pd
from matplotlib import style
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
style.use("ggplot")

# initialize
df = pd.read_csv('linear-regression.txt', header=None, sep=',')
df.columns=['x1','x2','y']
x,y,z = df['x1'],df['x2'],df['y']
rowLen=len(df['x1'])
df.insert(loc=0, column='x0',value=np.ones(rowLen))
df=df.T


dxy = df.iloc[0:3,:]
dxy =dxy.values
ddt=np.dot(dxy, dxy.T)
iddt=np.linalg.inv(ddt)
t1=np.dot(iddt,dxy)
dy= df.iloc[3:4,:].values
w=np.dot(t1,dy.T)
result = pd.DataFrame(data= w, index=['w0','w1','w2'])
print("\nthe Linear_ Regression result is" )
print(result)


