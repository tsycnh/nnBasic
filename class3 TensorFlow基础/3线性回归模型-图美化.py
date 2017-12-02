'''


1. name参数
2. namescope
'''
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



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
    # 保存图
    writer = tf.summary.FileWriter(logdir='logs',graph = tf.get_default_graph())
    writer.flush()# 强制写入文件并清空队列
