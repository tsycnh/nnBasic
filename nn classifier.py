import numpy as np
import csv
import funcs
import matplotlib.pyplot as plt
'''
Step1：载入数据
Step2：构建模型
       --非线性
Step3：编写loss计算
Step4：编写训练过程(反向传播)
Step5：开始训练
'''

# Step1：读取数据
with open('data2.txt','r') as my_file:# 知识点：with 上下文管理器 参考：https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/index.html
    # all_data = my_file.read()            # with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，比如文件使用后自动关闭、线程中锁的自动获取和释放等。
    # d = all_data.split('\n')
    # for x in d:
    #     x.split(',')
    csvdata = csv.reader(my_file)
    input_x = []
    input_y = []
    for line in csvdata:
        input_x.append([float(line[0]),float(line[1])])
        input_y.append([1,0] if int(line[2])==1 else [0,1]) # one hot 编码。类三目运算符 V1 if X else V2


def data_normlize(data):# data 为numpy array格式数据
    # 求每一列的最值
    d_max = data.max(axis=0)
    d_min = data.min(axis=0)
    data_normed = (data-d_min)/(d_max-d_min)
    return data_normed

input_x = np.array(input_x)
input_y = np.array(input_y)

# 数据尺度相差不大，也可不归一化，影响不大
input_x = data_normlize(input_x)
input_y = data_normlize(input_y)

# 可视化
# funcs.plot_data(input_x[:,0],input_x[:,1])
# plt.pause(444)

#Step2:构造模型
class BpModel:
    def __init__(self,x,y):
        self.x = x.transpose()
        self.y = y.transpose()
        self.w = np.random.rand(5,2)
        self.b = np.random.rand(5,1)

        self.w2,self.b2 = self.dense_layer(5,2)#  输出层

    def dense_layer(self,nodes_in,nodes_out):
        w = np.random.rand(nodes_out,nodes_in)
        b = np.random.rand(nodes_out,1)
        return w,b
    def forward(self):
        # tmp = self.w*self.x + self.b # 注意：numpy中的array相乘是逐个元素相乘
        net = np.dot(self.w,self.x) + self.b
        net = self.tanh(net)
        net = np.dot(self.w2,net) + self.b2
        net = self.softmax(net)
        return net

    # 激活函数
    def tanh(self,x):
        return (1-np.exp(-2*x))/(1+np.exp(-2*x))
    def softmax(self,x):
        ex = np.exp(x)
        s = ex.sum(axis= 0)
        soft = ex/s
        return soft

    # def d_tanh(self,x):



bp = BpModel(input_x,input_y)
result = bp.forward()
# print(result)

#Step 3：计算loss

# def calc_mse_loss(y,y_p):# 输入数据要为np.array格式
#     # 用MSE算法, 使用矩阵运算
#     tmp = y-y_p
#     tmp = tmp**2
#     allsum = np.sum(tmp)
#     data_size = np.size(tmp,1)
#     mse = allsum/(2*data_size)
#
#     return mse
def calc_cross_entropy_loss(y,y_p):
    log = np.log(y_p)
    tmp = y*log
    s = tmp[0,:]+tmp[1,:]
    ce = -np.sum(s)/len(s)
    return ce
loss = calc_cross_entropy_loss(bp.y,result)
print(loss)
foo = 666
