import tensorflow as tf

# 1.常量 tf.constant

a = tf.constant(2)
b = tf.constant(6)

# 启动默认计算图
sess = tf.Session()
# with tf.Session() as sess:
print('乘法:%i'%sess.run(a*b))
print('加法:%i'%sess.run(a+b))
sess.close()

# 2.变量 tf.Variable
# 2.占位符 tf.placeholder 特殊的变量
# 像上课占座一样

c = tf.placeholder(tf.float32)
d = tf.placeholder(tf.float32)

add = tf.add(c,d) # 一个operation
mul = tf.multiply(c,d)

sess = tf.Session()

result1 = sess.run(add,feed_dict={
    c:77,
    d:12
})

result2 = sess.run(mul,feed_dict={
    c:11,
    d:9
})
print('placeholder加法%f'%result1)
print('placeholder乘法%f'%result2)

sess.close()

# 3.矩阵相乘
# 创建一个 constant op，产生1*2矩阵
# 该op作为一个节点加入默认的计算图
# 返回值代表了constant op的输出
matrix1 = tf.constant([[3,9]])# 1*2
# 创建另一个constant op
matrix2 = tf.constant([[2],[7]])# 2*1
# 创建了一个matmul op
product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    # 调用run（product）会导致三个op节点的执行，1个matmul，2个constant
    r = sess.run(product)
    print('矩阵相乘',r)

# *4. 保存计算图

writer = tf.summary.FileWriter(logdir='logs',graph = tf.get_default_graph())
writer.flush()# 强制写入文件并清空队列

# *5. 查看tensorboard