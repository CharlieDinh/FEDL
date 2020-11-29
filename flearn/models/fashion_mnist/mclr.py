import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tqdm import trange

from flearn.utils.model_utils import batch_data, suffer_data, get_random_batch_sample
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad, prox_L2

class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''

    def __init__(self, num_classes, optimizer, seed=1):

        # params
        self.num_classes = num_classes

        self.optimizer = optimizer
        #self.vzero = 
        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(
                optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(
                self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def set_vzero(self, vzero):
        self.vzero = vzero

    def create_model(self, optimizer):
        """Model function for Logistic Regression."""
        features = tf.placeholder(
            tf.float32, shape=[None, 784], name='features')
        labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        logits = tf.layers.dense(inputs=features, units=self.num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(
            0.1))  # 0.001  #Linear layer without regularizer
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(
            tf.equal(labels, predictions["classes"]))
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

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads

    def get_raw_gradients(self, data):
        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], self.labels: data['y']})
        return model_grads

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
        if(optimizer == "fedprox"):
            data_x, data_y = suffer_data(data)
            for _ in range(num_epochs):  # t = 1,2,3,4,5,...m
                X, y = get_random_batch_sample(data_x, data_y, batch_size)
                with self.graph.as_default():
                    self.sess.run(self.train_op, feed_dict={
                        self.features: X, self.labels: y})
        else:
            wzero = self.get_params()
            w1 = wzero - self.optimizer._lr * np.array(self.vzero)
            w1 = prox_L2(np.array(w1), np.array(wzero),
                         self.optimizer._lr, self.optimizer._lamb)
            self.set_params(w1)

            for _ in range(num_epochs):  # t = 1,2,3,4,5,...m
                X, y = get_random_batch_sample(data_x, data_y, batch_size)
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
                    else:   # fedsgd or fedprox
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
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()
