import numpy as np
import tensorflow as tf
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

    def set_gradientParam(self, preG, preGn):
        self.optimizer.set_preG(preG, self)
        self.optimizer.set_preGn(preGn, self)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], self.labels: data['y']})

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
            batch_size = len(data['y'])#//10

        #if(optimizer == "fedavg"):
        #data_x, data_y = suffer_data(data)
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            #X, y = get_random_batch_sample(data_x, data_y, batch_size)
            #with self.graph.as_default():
            #    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        with self.graph.as_default():
            grad = self.sess.run(self.grads, feed_dict={self.features: data['x'], self.labels: data['y']})
        comp = num_epochs * \
            (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, grad, comp

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
