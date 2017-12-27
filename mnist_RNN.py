#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:31:35 2017

@author: alex
"""
import tensorflow as tf
import os
import struct
import numpy as np
n_batch=1000
batch_size=128
n_steps=28
dim=28
n_class=10
n_units=128
tf.reset_default_graph()
def get_cell():
    return tf.nn.rnn_cell.BasicLSTMCell(n_units)
def get_batch(batch,data_in,data_out):
    images=np.zeros((batch,n_steps*dim))
    labels=np.zeros((batch,n_class))
    for i in range(batch):
        ran=np.random.randint(0,data_in.shape[0])
        images[i]=data_in[ran]
        labels[i]=data_out[ran]
    images=np.reshape(images,(batch,n_steps,dim))           
    return images,labels
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join('./', '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join('./', '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
def compute_acc():
    prediction=sess.run(output,feed_dict={x:images,y:labels})
    n=0
    for i in range(batch_size):
        if np.argmax(prediction[i])==np.argmax(labels[i]):
            n=n+1
    return n/batch_size
x_train, y_train = load_mnist('../data', kind='train')
print('Rows: %d, columns: %d' % (x_train.shape[0], x_train.shape[1]))
x_test, y_test = load_mnist('../data', kind='t10k')
print('Rows: %d, columns: %d' % (x_test.shape[0], x_test.shape[1]))

y_train_ohe=np.zeros((y_train.shape[0],n_class))
y_test_ohe=np.zeros((y_test.shape[0],n_class))
for i in range(y_train.shape[0]):
    y_train_ohe[i,y_train[i]]=1
for i in range(y_test.shape[0]):
    y_test_ohe[i,y_test[i]]=1



with tf.name_scope('inputs'):
    x=tf.placeholder(tf.float32,([None,n_steps,dim]))
    y=tf.placeholder(tf.float32,([None,n_class]))

cell=tf.nn.rnn_cell.MultiRNNCell([get_cell() for i in range(2)])
with tf.name_scope('lstm'):
    outputs,state=tf.nn.dynamic_rnn(cell,x,initial_state=None,dtype=tf.float32)
with tf.name_scope('softmax'):
    output=tf.layers.dense(outputs[:,-1,:],10,activation=tf.nn.softmax)
with tf.name_scope('loss'):
    loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(output),reduction_indices=[1]))
with tf.name_scope('optimizer'):
    optimizer=tf.train.AdamOptimizer().minimize(loss)
with tf.name_scope('acc'):
    acc=tf.metrics.accuracy(tf.argmax(y,axis=1),tf.argmax(output,axis=1),[1])

init=tf.global_variables_initializer()


sess=tf.Session()
sess.run(init)

merged=tf.summary.merge_all
writer=tf.summary.FileWriter('./logs',sess.graph)

for i in range(1000):
    images,labels=get_batch(batch_size,x_train,y_train_ohe)
    sess.run(optimizer,feed_dict={x:images,y:labels})
    if i%100==0:
        print('acc=',compute_acc())

