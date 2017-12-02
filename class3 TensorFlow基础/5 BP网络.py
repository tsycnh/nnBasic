import csv
import numpy as np
import tensorflow as tf

def load_data(filepath):

    with open(filepath,'r') as my_file:
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

input_x,input_y = load_data('data2.txt')
test_x,test_y = load_data('data2_test.txt')

def full_layer(input,nodes_in,nodes_out):
    w = tf.Variable(tf.truncated_normal([nodes_in,nodes_out],stddev=0.1))
    b = tf.Variable(tf.truncated_normal([1,nodes_out],stddev=0.1))

    tmp = tf.matmul(input,w)+b
    output = tf.nn.tanh(tmp)
    return output
def full_layer2(input,nodes_in,nodes_out):
    w = tf.Variable(tf.truncated_normal([nodes_in,nodes_out],stddev=0.1))
    b = tf.Variable(tf.truncated_normal([1,nodes_out],stddev=0.1))

    tmp = tf.matmul(input,w)+b
    # output = tf.nn.softmax(tmp)
    return tmp
with tf.Graph().as_default():
    X = tf.placeholder(tf.float32,shape=(200,2),name='X')
    Y = tf.placeholder(tf.float32,shape=(200,2),name='Y')

    #结构
    # net = full_layer(X,2,5)
    net = tf.layers.dense(X,5,activation=tf.nn.tanh)
    predicts = full_layer2(net,5,2)
    # loss cross entropy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicts,labels=Y))

    optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optim.minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        sess.run(train_op,feed_dict={X:input_x,Y:input_y})
        if step%100 == 0:
            print('loss:',sess.run(loss,feed_dict={X:input_x,Y:input_y}))

    print('training finish')
    sess.close()