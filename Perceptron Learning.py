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
import matplotlib.pyplot as plt
style.use("ggplot")

# initialize
df = pd.read_csv('classification.txt', header=None, sep=',')
df = df.iloc[:,0:4]
df.columns=['x1','x2','x3','r']
rowLen=len(df['x1'])
df.insert(loc=0, column='x0',value=pd.Series(np.ones((rowLen))))
w=np.random.random([1,4])
# set a = 0.000000000000001
a = 0.001

### Perceptron Learning algorithm
violet=True
pla=[]


while violet :
    violet = False
    numv = 0
    for index, point in df.iterrows():
        temp= np.dot(point.iloc[0:4],w.T)
        if temp * point['r']<0 :
            violet =True
            numv += 1
            if point['r']<0 :
                w= np.subtract(w,(a*point.iloc[0:4]).values.reshape(1,4))
            else :
                w= np.add(w,(a*point.iloc[0:4]).values.reshape(1,4))
    pla.append(numv)

print("this is the Perceptron Learning algorithm result :")
plaRes= pd.DataFrame(data=w,columns=['w0','w1','w2','w3'])
print(plaRes)

axisX = range(len(pla))
fig = plt.figure()
ax1 = fig.add_subplot(111)
#设置标题
ax1.set_title('Perceptron Learning figure')
#设置X轴标签
plt.xlabel('Times')
#设置Y轴标签
plt.ylabel('violet points')
#画散点图
ax1.scatter(axisX,pla,c = 'r',marker = '.')
#设置图标
plt.legend('x')
#显示所画的图
plt.show()
