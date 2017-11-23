import numpy as np

# 生成买桔子任务用数据集
xx = np.random.random([100,])*5
yy = 3*xx + np.random.random([100,])
print('xx',xx)
x = xx
y = yy
datafile = open('data.txt',mode='w')
all_data = ''
for i in range(len(xx)):
    linedata = str(xx[i])+','+str(yy[i])+'\n'
    all_data += linedata

datafile.write(all_data)
datafile.close()

