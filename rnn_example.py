# coding=gbk

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim=10):
        # Hyperparameters
        self.input_dim = input_dim #输入维度,一次只输入一个
        '''
        每个时序的LSTM单元都是(1, 10, 1)
        输入一个值(input_dim=1),经过训练之后会得到一个值,而这个值,会参与到下一个时序的训练中
        下一次时序也输入一个值,那么这个值代表本时序,而从上一个时序传过来的结果就代表上一个时序
        所以RNN网络才能拥有'记忆能力'
        而LSTM网络又是RNN网络的改进版,加入忘记门,不想要的记忆可以通过忘记门参数忘记
        '''
        self.seq_size = seq_size #分为四个阶段输入,最后得出的结果也是4个
        self.hidden_dim = hidden_dim #隐层神经元10个

        # Weight variables and input placeholders
        '''
        每一个样本,每个时序传入1个值,这一个值经过10个隐层神经元的LSTM网络训练得出的应该是1个值
        所以这里的W_out和b_out就是输出层的参数,shape是(10, 1)和(1)
        '''
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out') #W是(10, 1)
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out') #b是(1)
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim]) #(样本数, 4, 1)
        self.y = tf.placeholder(tf.float32, [None, seq_size]) #(样本数, 4)

        # Cost optimizer
        '''
        误差使用均方误差
        '''
        self.cost = tf.reduce_mean(tf.square((self.model() - self.y) ** 2))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver() #保存模型

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim) #使用隐藏层单元数10,来创建LSTM网络的cell单元
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32) #对cell进行堆叠
        num_examples = tf.shape(self.x)[0] #样本数量
        # tf.expand_dims,增加一个新的维度
        # W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])

        '''
        在0的位置增加一个维度,W_out从(10, 1)到(1, 10, 1)
        '''
        tf_expand = tf.expand_dims(self.W_out, 0)
        '''
        tile前是(1, 10, 1),tile后是(?, 10, 1)
        做tile操作是为了3个样本并行计算
        '''
        tf_tile = tf.tile(tf_expand, [num_examples, 1, 1]) #(?, 10, 1)
        '''
        结果out是(?, 4, 1)
        不管有多少样本,都可以得到4个数字的输出值
        '''
        out = tf.matmul(outputs, tf_tile) + self.b_out #output是(?, 4, 10), tf_tile是(?, 10, 1)
        # tf.squeeze 删除所有维度是1的
        out = tf.squeeze(out)
        return out #返回一个结果op(?, 4)

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables() #重复使用之前的变量作用域
            sess.run(tf.global_variables_initializer())
            for i in range(3000):
                _, mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print(i, mse) #迭代次数和误差
            save_path = self.saver.save(sess, './model/simple_rnn') #存储模型
            print('Model saved to {}'.format(save_path))

    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables() #重复使用之前的变量作用域
            self.saver.restore(sess, './model/simple_rnn') #加载模型
            output = sess.run(self.model(), feed_dict={self.x: test_x})
            return output


if __name__ == '__main__':
    predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=20)
    '''
    预测值是本次和上一次的值之和
    既然rnn网络能用来记忆之前的值,那么我们看看使用这样的数据进行训练之后,测试的效果如何
    
    数据是三维的,意味着训练集一共有三个样本
    而每个样本都是4个数字(二维)
    我们一次送进LSTM网络1个数字,总共送4次,才把一个样本送进去
    '''
    train_x = [[[1], [2], [5], [6]],
               [[5], [7], [7], [8]],
               [[3], [4], [5], [7]]]
    train_y = [[1, 3, 7, 11],
               [5, 12, 14, 15],
               [3, 7, 9, 12]]
    #训练
    predictor.train(train_x, train_y)

    #测试集
    test_x = [[[1], [2], [3], [4]],  # 1, 3, 5, 7
              [[4], [5], [6], [7]]]  # 4, 9, 11, 13
    actual_y = [[[1], [3], [5], [7]],
                [[4], [9], [11], [13]]]
    #预测结果
    pred_y = predictor.test(test_x)

    print("\nLets run some tests!\n")

    for i, x in enumerate(test_x):
        print("When the input is {}".format(x))
        print("The ground truth output should be {}".format(actual_y[i]))
        print("And the model thinks it is {}\n".format(pred_y[i]))