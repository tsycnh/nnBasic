import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

# 生成桔子分类任务用数据集

np.random.seed(0)
x, y = sklearn.datasets.make_moons(400, noise=0.20)


datafile = open('data2_test.txt',mode='w')
all_data = ''
for i in range(len(x)):
    linedata = str(x[i][0]) + ',' + str(x[i][1]) + ',' + str(y[i]) + '\n'
    all_data += linedata

datafile.write(all_data)
datafile.close()

x1 = x[:,0]
x2 = x[:,1]
plt.scatter(x[:, 0], x[:, 1], s=20, c=y, cmap=plt.cm.Spectral)
plt.pause(666)

