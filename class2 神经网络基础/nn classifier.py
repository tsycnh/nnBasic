import numpy as np
import csv
import funcs
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

'''
Step1：载入数据
Step2：构建模型
       --非线性
Step3：编写loss计算,使用cross entropy
Step4：编写训练过程(反向传播)
Step5：开始训练
'''

# Step1：读取数据
def load_data(filepath):

    with open(filepath,'r') as my_file:# 知识点：with 上下文管理器 参考：https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/index.html
        # all_data = my_file.read()            # with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，比如文件使用后自动关闭、线程中锁的自动获取和释放等。
        # d = all_data.split('\n')
        # for x in d:
        #     x.split(',')
        csvdata = csv.reader(my_file)
        input_x = []
        input_y = []
        origin_y = []
        for line in csvdata:
            input_x.append([float(line[0]),float(line[1])])
            origin_y.append(int(line[2]))
            input_y.append([1,0] if int(line[2])==1 else [0,1]) # one hot 编码。类三目运算符 V1 if X else V2
    input_x = np.array(input_x)
    input_y = np.array(input_y)
    return input_x,input_y

def data_normlize(data):# data 为numpy array格式数据
    # 求每一列的最值
    d_max = data.max(axis=0)
    d_min = data.min(axis=0)
    data_normed = (data-d_min)/(d_max-d_min)
    return data_normed



# # 数据尺度相差不大，也可不归一化，影响不大
# input_x = data_normlize(input_x)
# input_y = data_normlize(input_y)



#Step2:构造模型
class BpModel:

    def __init__(self,x,y):
        self.x = x # x无需转置。每行一个样本，一共200行
        self.y = y # y 每行为一个样本的one-hot编码，一共200行
        self.w = np.random.rand(2,5)# 权重w按in*out大小初始化
        self.b = np.random.rand(1,5)# 偏置b按1*out大小初始化

        self.w2,self.b2 = self.dense_layer(5,2)#  输出层
        self.alpha = 0.01
    def dense_layer(self,nodes_in,nodes_out):
        w = np.random.rand(nodes_in,nodes_out)
        b = np.random.rand(1,nodes_out)
        return w,b
    def forward(self):
        # tmp = self.w*self.x + self.b # 注意：numpy中的array相乘是逐个元素相乘（汉密尔顿积）
        z1 = np.dot(self.x,self.w) + self.b
        a1 = self.tanh(z1)
        z2 = np.dot(a1,self.w2)
        z2 += self.b2
        probs = self.softmax(z2)# 分类问题的输出层

        self.z1 = z1
        self.a1 = a1
        self.z2 = z2
        self.probs = probs
        return probs

    # 激活函数
    def tanh(self,x):
        return (1-np.exp(-2*x))/(1+np.exp(-2*x))
    def softmax(self,x):
        ex = np.exp(x)
        s = ex.sum(axis= 1,keepdims=True)
        soft = ex/(s+0.000001)
        return soft

    def backward(self):
        delta3 =self.probs- self.y
        d_w2 =np.dot(self.a1.T,delta3)
        d_b2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3,self.w2.T) * (1 - np.power(self.a1, 2))
        d_w1 = np.dot(self.x.T,delta2)
        d_b1 = np.sum(delta2, axis=0,keepdims=True)


        self.w2 = self.w2 - self.alpha*d_w2
        self.b2 = self.b2 - self.alpha*d_b2
        self.w = self.w - self.alpha*d_w1
        self.b = self.b - self.alpha*d_b1

    def predict(self,x,y):
        self.x = x
        self.y = y
        result = self.forward()

        for i,d in enumerate(result):
            for j,c in enumerate(d):
                if result[i][j]>=0.5:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        match = 0
        not_match = 0
        for i,d in enumerate(result):
                if (result[i] == y[i]).all():
                    match+=1
                else:
                    not_match+=1

        print('match:',match,'not match:',not_match,'accuracy:',match/(match+not_match))

input_x,input_y = load_data('data2.txt')
test_data_x,test_data_y = load_data('data2_test.txt')
bp = BpModel(input_x,input_y)
# print(result)
# 可视化
# funcs.plot_data(input_x[:,0],input_x[:,1])
# plt.pause(444)
#Step 3：计算loss

def calc_cross_entropy_loss(y,y_p):
    log = np.log(y_p+0.000001)
    tmp = y*log
    s = tmp[:,0]+tmp[:,1]
    ce = -np.sum(s)/len(s)
    return ce

for step in range(0,1000):
    result = bp.forward()
    loss = calc_cross_entropy_loss(bp.y, result)#(200*2)
    bp.backward()
    if step%100 == 0:
        print('step:',step,'loss:',loss)

bp.predict(input_x,input_y)
bp.predict(test_data_x,test_data_y)