# coding=gbk

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim=10):
        # Hyperparameters
        self.input_dim = input_dim #����ά��,һ��ֻ����һ��
        '''
        ÿ��ʱ���LSTM��Ԫ����(1, 10, 1)
        ����һ��ֵ(input_dim=1),����ѵ��֮���õ�һ��ֵ,�����ֵ,����뵽��һ��ʱ���ѵ����
        ��һ��ʱ��Ҳ����һ��ֵ,��ô���ֵ����ʱ��,������һ��ʱ�򴫹����Ľ���ʹ�����һ��ʱ��
        ����RNN�������ӵ��'��������'
        ��LSTM��������RNN����ĸĽ���,����������,����Ҫ�ļ������ͨ�������Ų�������
        '''
        self.seq_size = seq_size #��Ϊ�ĸ��׶�����,���ó��Ľ��Ҳ��4��
        self.hidden_dim = hidden_dim #������Ԫ10��

        # Weight variables and input placeholders
        '''
        ÿһ������,ÿ��ʱ����1��ֵ,��һ��ֵ����10��������Ԫ��LSTM����ѵ���ó���Ӧ����1��ֵ
        ���������W_out��b_out���������Ĳ���,shape��(10, 1)��(1)
        '''
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out') #W��(10, 1)
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out') #b��(1)
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim]) #(������, 4, 1)
        self.y = tf.placeholder(tf.float32, [None, seq_size]) #(������, 4)

        # Cost optimizer
        '''
        ���ʹ�þ������
        '''
        self.cost = tf.reduce_mean(tf.square((self.model() - self.y) ** 2))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver() #����ģ��

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim) #ʹ�����ز㵥Ԫ��10,������LSTM�����cell��Ԫ
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32) #��cell���жѵ�
        num_examples = tf.shape(self.x)[0] #��������
        # tf.expand_dims,����һ���µ�ά��
        # W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])

        '''
        ��0��λ������һ��ά��,W_out��(10, 1)��(1, 10, 1)
        '''
        tf_expand = tf.expand_dims(self.W_out, 0)
        '''
        tileǰ��(1, 10, 1),tile����(?, 10, 1)
        ��tile������Ϊ��3���������м���
        '''
        tf_tile = tf.tile(tf_expand, [num_examples, 1, 1]) #(?, 10, 1)
        '''
        ���out��(?, 4, 1)
        �����ж�������,�����Եõ�4�����ֵ����ֵ
        '''
        out = tf.matmul(outputs, tf_tile) + self.b_out #output��(?, 4, 10), tf_tile��(?, 10, 1)
        # tf.squeeze ɾ������ά����1��
        out = tf.squeeze(out)
        return out #����һ�����op(?, 4)

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables() #�ظ�ʹ��֮ǰ�ı���������
            sess.run(tf.global_variables_initializer())
            for i in range(3000):
                _, mse = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print(i, mse) #�������������
            save_path = self.saver.save(sess, './model/simple_rnn') #�洢ģ��
            print('Model saved to {}'.format(save_path))

    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables() #�ظ�ʹ��֮ǰ�ı���������
            self.saver.restore(sess, './model/simple_rnn') #����ģ��
            output = sess.run(self.model(), feed_dict={self.x: test_x})
            return output


if __name__ == '__main__':
    predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=20)
    '''
    Ԥ��ֵ�Ǳ��κ���һ�ε�ֵ֮��
    ��Ȼrnn��������������֮ǰ��ֵ,��ô���ǿ���ʹ�����������ݽ���ѵ��֮��,���Ե�Ч�����
    
    ��������ά��,��ζ��ѵ����һ������������
    ��ÿ����������4������(��ά)
    ����һ���ͽ�LSTM����1������,�ܹ���4��,�Ű�һ�������ͽ�ȥ
    '''
    train_x = [[[1], [2], [5], [6]],
               [[5], [7], [7], [8]],
               [[3], [4], [5], [7]]]
    train_y = [[1, 3, 7, 11],
               [5, 12, 14, 15],
               [3, 7, 9, 12]]
    #ѵ��
    predictor.train(train_x, train_y)

    #���Լ�
    test_x = [[[1], [2], [3], [4]],  # 1, 3, 5, 7
              [[4], [5], [6], [7]]]  # 4, 9, 11, 13
    actual_y = [[[1], [3], [5], [7]],
                [[4], [9], [11], [13]]]
    #Ԥ����
    pred_y = predictor.test(test_x)

    print("\nLets run some tests!\n")

    for i, x in enumerate(test_x):
        print("When the input is {}".format(x))
        print("The ground truth output should be {}".format(actual_y[i]))
        print("And the model thinks it is {}\n".format(pred_y[i]))