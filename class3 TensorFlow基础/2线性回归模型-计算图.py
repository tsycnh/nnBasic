'''

Step1: 设计构造图
'''
import tensorflow as tf


with tf.Graph().as_default():# 默认构造图
    # 输入占位符
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    # 模型参数变量
    W = tf.Variable(tf.zeros([1]))
    b = tf.Variable(tf.zeros([1]))
    # 构建模型
    Y_p = tf.add(tf.multiply(X,W),b)

    # 计算loss MSE
    loss = tf.reduce_mean(tf.pow((Y-Y_p),2))/2

    # 优化函数
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optim.minimize(loss) # 自动求导

    # 评估 ：评价参数可能和训练时不同。回归问题相同
    EvalLoss = tf.reduce_mean(tf.pow((Y-Y_p),2))/2
    # 保存图
    writer = tf.summary.FileWriter(logdir='logs',graph = tf.get_default_graph())
    writer.flush()# 强制写入文件并清空队列
