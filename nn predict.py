import numpy as np
import matplotlib.pyplot as plt
import csv
import funcs
import random
'''
Step1：载入数据
       --数据归一化
Step2：构建模型
Step3：编写loss计算
Step4：编写训练过程(反向传播)
Step5：开始训练
'''
# Step1：读取数据
with open('data.txt','r') as my_file:# 知识点：with 上下文管理器 参考：https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/index.html
    # all_data = my_file.read()            # with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，比如文件使用后自动关闭、线程中锁的自动获取和释放等。
    # d = all_data.split('\n')
    # for x in d:
    #     x.split(',')
    csvdata = csv.reader(my_file)
    input_x = []
    input_y = []
    for line in csvdata:
        input_x.append(float(line[0]))
        input_y.append(float(line[1]))

# 数据归一化
# x_max = max(input_x)
# x_min = min(input_x)
#
# for i,x in enumerate(input_x):
#
#     input_x[i] = (x-x_min)/(x_max-x_min)
#
def data_normlize(data):# in-place 操作
    d_max = max(data)
    d_min = min(data)
    for i, d in enumerate(data):
        data[i] = (d - d_min) / (d_max - d_min)

data_normlize(input_x)
data_normlize(input_y)

input_x = np.array(input_x)
input_y = np.array(input_y)

# Step2：构建模型
# funcs.plot_data(input_x,input_y)# 可视化

class LModel:# 全批次计算
    def __init__(self,x,y):
        self.w = random.random()
        self.b = random.random()
        self.x = x
        self.y = y
        self.deltaw = 0.0
        self.deltab = 0.0
        self.alpha = 0.01
        self.batch_size = len(self.x)
    def forward(self):
        y_p = self.w*self.x + self.b
        return y_p

    def calc_delta_w(self):
        tmp = (self.w*self.x+self.b -self.y)*self.x
        self.deltaw = sum(tmp)/self.batch_size
    def calc_delta_b(self):
        tmp = (self.w*self.x+self.b -self.y)
        self.deltab = sum(tmp)/self.batch_size
    def update(self):
        self.w = self.w - self.alpha*self.deltaw
        self.b = self.b - self.alpha*self.deltab

#Step3: 写loss

def calc_loss(y,y_p):# 输入数据要为np.array格式
    # 用MSE算法, 使用矩阵运算
    tmp = y-y_p
    tmp = tmp**2
    allsum = sum(tmp)
    mse = allsum/(2*len(y))

    return mse

# loss = calc_loss(input_x,input_y)
# Step4: 参数更新（类内）

# def delta_w

a_model = LModel(x=input_x,y=input_y)
print(a_model.w,a_model.b)

# Step5：开始训练
for step in range(0,5000):
    y_p = a_model.forward()# 计算预测值
    loss = calc_loss(y=input_y,y_p=y_p)
    a_model.calc_delta_w()
    a_model.calc_delta_b()
    a_model.update()

    #可视化结果
    if step%10 == 0:
        print('step:', step, 'loss:', loss, 'w:', a_model.w, 'b:', a_model.b)
        funcs.plot_data(input_x,input_y)
        funcs.plot_line(a_model.w,a_model.b)
        plt.pause(0.01)