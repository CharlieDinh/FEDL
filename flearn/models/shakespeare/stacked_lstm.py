import numpy as np
from tqdm import trange
import json

import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn

utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)

from flearn.utils.model_utils import batch_data, get_random_batch_sample, suffer_data
from flearn.utils.language_utils import letter_to_vec, word_to_indices
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_sparse_grad,prox_L2

def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch

class Model(object):
    def __init__(self, seq_len, num_classes, n_hidden, optimizer, seed):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.optimizer = optimizer
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer):
        features = tf.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.get_variable("embedding", [self.num_classes, 8])
        x = tf.nn.embedding_lookup(embedding, features)
        labels = tf.placeholder(tf.int32, [None, self.num_classes])
        
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        pred = tf.layers.dense(inputs=outputs[:,-1,:], units=self.num_classes)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())


        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, grads, eval_metric_ops, loss


    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        '''in order to avoid the OOM error, we need to calculate the gradients on each 
        client batch by batch. batch size here is set to be 100.

        Return: a one-D array (after flattening all gradients)
        '''
        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        processed_samples = 0

        if num_samples < 50:
            input_data = process_x(data['x'])
            target_data = process_y(data['y'])
            with self.graph.as_default():
                model_grads = self.sess.run(self.grads, 
                    feed_dict={self.features: input_data, self.labels: target_data})
            grads = process_sparse_grad(model_grads)
            processed_samples = num_samples

        else:  # in order to fit into memory, compute gradients in a batch of size 50, and subsample a subset of points to approximate

            for i in range(min(int(num_samples / 50), 4)):
                input_data = process_x(data['x'][50*i:50*(i+1)])
                target_data = process_y(data['y'][50*i:50*(i+1)])

                with self.graph.as_default():
                    model_grads = self.sess.run(self.grads,
                    feed_dict={self.features: input_data, self.labels: target_data})
            
                flat_grad = process_sparse_grad(model_grads)
                grads = np.add(grads, flat_grad)

            grads = grads * 1.0 / min(int(num_samples/50), 4)
            processed_samples = min(int(num_samples / 50), 4) * 50

        return processed_samples, grads

    def get_raw_gradients(self, data):

        num_samples = len(data['y'])
        if num_samples < 50:
            input_data = process_x(data['x'])
            target_data = process_y(data['y'])
            with self.graph.as_default():
                model_grads = self.sess.run(self.grads,
                                            feed_dict={self.features: input_data, self.labels: target_data})

        else:  # calculate the grads in a batch size of 50
            for i in range(min(int(num_samples / 50), 4)):
                input_data = process_x(data['x'][50*i:50*(i+1)])
                target_data = process_y(data['y'][50*i:50*(i+1)])
                with self.graph.as_default():
                    model_grads = self.sess.run(self.grads,
                                                feed_dict={self.features: input_data, self.labels: target_data})

        return model_grads


    def set_vzero(self, vzero):
        self.vzero = vzero

    def solve_inner(self, optimizer, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        if (batch_size == 0):  # Full data or batch_size
            # print("Full dataset")
            batch_size = len(data['y'])

        if(optimizer == "fedavg"):
            for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
                for X, y in batch_data(data, batch_size):
                    with self.graph.as_default():
                        self.sess.run(self.train_op, feed_dict={
                                      self.features: X, self.labels: y})
        else:
            wzero = self.get_params()
            data_x, data_y = suffer_data(data)
            w1 = wzero - self.optimizer._lr * np.array(self.vzero)
            w1 = prox_L2(np.array(w1), np.array(wzero),
                         self.optimizer._lr, self.optimizer._lamb)
            self.set_params(w1)

            for _ in range(num_epochs):  # t = 1,2,3,4,5,...m
                X, y = get_random_batch_sample(data_x, data_y, batch_size)
                X = process_x(X)
                y = process_y(y)
                with self.graph.as_default():
                    # get the current weight
                    if(optimizer == "fedsvrg"):
                        current_weight = self.get_params()

                        # calculate fw0 first:
                        self.set_params(wzero)
                        fwzero = self.sess.run(self.grads, feed_dict={
                                               self.features: X, self.labels: y})
                        self.optimizer.set_fwzero(fwzero, self)

                        # return the current weight to the model
                        self.set_params(current_weight)
                        self.sess.run(self.train_op, feed_dict={
                            self.features: X, self.labels: y})
                    elif(optimizer == "fedsarah"):
                        if(_ == 0):
                            self.set_params(wzero)
                            grad_w0 = self.sess.run(self.grads, feed_dict={
                                                    self.features: X, self.labels: y})  # grad w0)
                            self.optimizer.set_preG(grad_w0, self)
                            self.set_params(w1)
                            preW = self.get_params()   # previous is w1
                            self.sess.run(self.train_op, feed_dict={
                                self.features: X, self.labels: y})
                        else:
                         # == w1
                            curW = self.get_params()

                            # get previous grad
                            self.set_params(preW)
                            grad_preW = self.sess.run(self.grads, feed_dict={
                                                      self.features: X, self.labels: y})  # grad w0)
                            self.optimizer.set_preG(grad_preW, self)
                            preW = curW
                            # return back curent grad
                            self.set_params(curW)
                            self.sess.run(self.train_op, feed_dict={
                                          self.features: X, self.labels: y})
                    else:   # fedsgd
                        self.sess.run(self.train_op, feed_dict={
                                      self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * \
            (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            tot_correct: total #samples that are predicted correctly
            loss: loss value on `data`
        '''
        x_vecs = process_x(data['x'])
        labels = process_y(data['y'])
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()

