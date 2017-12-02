'''
1. 初始化节点
2. 启动会话
'''
import tensorflow as tf
import csv
import numpy as np
# 数据读取
with open('data.txt','r') as my_file:
    csvdata = csv.reader(my_file)
    input_x = []
    input_y = []
    for line in csvdata:
        input_x.append(float(line[0]))
        input_y.append(float(line[1]))

def data_normlize(data):
    d_max = max(data)
    d_min = min(data)
    for i, d in enumerate(data):
        data[i] = (d - d_min) / (d_max - d_min)

data_normlize(input_x)
data_normlize(input_y)

input_x = np.array(input_x)
input_y = np.array(input_y)
#===
with open('data_test.txt','r') as my_file:
    csvdata = csv.reader(my_file)
    test_x = []
    test_y = []
    for line in csvdata:
        test_x.append(float(line[0]))
        test_y.append(float(line[1]))
data_normlize(test_x)
data_normlize(test_y)

test_x = np.array(test_x)
test_y = np.array(test_y)
#=======

with tf.Graph().as_default():# 默认构造图
    # 输入占位符
    with tf.name_scope('inputs'):
        X = tf.placeholder(tf.float32,name='X')# 参数name
        Y = tf.placeholder(tf.float32,name='Y')
    # 模型参数变量
    with tf.name_scope('forward'):
        W = tf.Variable(tf.zeros([1]),name='Weights')
        b = tf.Variable(tf.zeros([1]),name='bias')
        # 构建模型
        Y_p = tf.add(tf.multiply(X,W),b)

    # 计算loss MSE
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.pow((Y-Y_p),2))/2

    # 优化函数
    with tf.name_scope('train'):
        optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optim.minimize(loss) # 自动求导

    # 评估 ：评价参数可能和训练时不同。回归问题相同
    with tf.name_scope('evaluate'):
        EvalLoss = tf.reduce_mean(tf.pow((Y-Y_p),2))/2
    #-------
    # 初始化节点
    init_op = tf.global_variables_initializer()
    # 保存图
    tf.summary.scalar('train loss',loss)
    tf.summary.scalar('test loss',EvalLoss)
    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir='logs',graph = tf.get_default_graph())
    writer.flush()# 强制写入文件并清空队列
    # training 开启会话
    sess = tf.Session()
    sess.run(init_op)
    for step in range(0,1000):
        # for tx,ty in zip(input_x,input_y):# 一组一组的取 SGD
        #     _, train_loss, train_w, train_b = sess.run([train_op, loss, W, b], feed_dict={X: tx, Y: ty})

        _,train_loss,train_w,train_b = sess.run([train_op,loss,W,b],feed_dict={X:input_x,Y:input_y})
        if step%10 == 0:
            print('step:',step,'loss:',train_loss,'w:',train_w,'b:',train_b)

            test_loss = sess.run([EvalLoss],feed_dict={X:test_x,Y:test_y})
            print('test loss:',test_loss)
            s = sess.run(merge_summary,feed_dict={X:input_x,Y:input_y})
            writer.add_summary(s,step)

    print('training finish')