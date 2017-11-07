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
df = pd.read_csv('classification.txt', header=None, sep=',')
df.columns=['x1','x2','x3','rd','r']
del df['rd']
rowLen=len(df['x1'])
df.insert(loc=0, column='x0',value=pd.Series(np.ones((rowLen))))
w=np.random.random([4,1])
# set a = 0.000000000000001
a = 0.005



#big=df[df['r']>0]
#small=df[df['r']<0]
#draw= df.iloc[:,0:4]
#ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程
#ax.scatter(big.x1,big.x2,big.x3,c='r',s=1,) #绘制数据点
#ax.scatter(small.x1,small.x2,small.x3,c='g',s=1,) #绘制数据点


times=7000
i=0
dEin=np.zeros((4,1))
num = 0
lr = []
while i < times:
    i+=1
    num = 0
    dEin = np.zeros((4,1))
    for index, point in df.iterrows():
        s = np.dot(point.iloc[0:4], w)
        poss=np.exp(s) / (np.exp(s)+1)
        if (poss-0.5)*point['r'] < 0 :
            num+=1
        temp1=np.reciprocal(1+math.exp(point['r']*(np.dot(point.iloc[0:4],w))))*point['r']*point.iloc[0:4]
        dEin = np.add(dEin,temp1.values.reshape(4,1))
    dEin = dEin * (-1/rowLen)
    w= w - a*dEin
    lr.append(num)

print("this is the Logisitic Regression algorithm result :")
dw = pd.DataFrame(data=w,index=['w0','w1','w2','w3'])
print(dw)




axisX = range(len(lr))
fig = plt.figure()
ax1 = fig.add_subplot(111)
#设置标题
ax1.set_title('Logistic Learning figure')
#设置X轴标签
plt.xlabel('Times')
#设置Y轴标签
plt.ylabel('violet points')
#画散点图
ax1.scatter(axisX,lr,c = 'r',marker = '.')
#设置图标
plt.legend('x')
#显示所画的图
plt.show()
#ax.scatter3D(x,y,z)

print("finish")