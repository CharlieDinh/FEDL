import json
import numpy as np
import tensorflow as tf
from tqdm import trange

from tensorflow.contrib import rnn

from flearn.utils.model_utils import batch_data, get_random_batch_sample, suffer_data
from flearn.utils.language_utils import line_to_indices
from flearn.utils.tf_utils import graph_size, process_grad, prox_L2

with open('flearn/models/sent140/embs.json', 'r') as inf:
    embs = json.load(inf)
id2word = embs['vocab']
word2id = {v: k for k,v in enumerate(id2word)}
word_emb = np.array(embs['emba'])

def process_x(raw_x_batch, max_words=25):
    x_batch = [e[4] for e in raw_x_batch]
    x_batch = [line_to_indices(e, word2id, max_words) for e in x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [1 if e=='4' else 0 for e in raw_y_batch]
    y_batch = np.array(y_batch)

    return y_batch

class Model(object):

    def __init__(self, seq_len, num_classes, n_hidden, optimizer, seed):
        #params
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.emb_arr = word_emb
        self.optimizer = optimizer
        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    
    def create_model(self, optimizer):
        features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
        labels = tf.placeholder(tf.int64, [None,], name='labels')

        embs = tf.Variable(self.emb_arr, dtype=tf.float32, trainable=False)
        x = tf.nn.embedding_lookup(embs, features)
        
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        fc1 = tf.layers.dense(inputs=outputs[:,-1,:], units=30)
        pred = tf.squeeze(tf.layers.dense(inputs=fc1, units=1))
        
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=pred)
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        
        correct_pred = tf.equal(tf.to_int64(tf.greater(pred,0)), labels)
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

    def get_gradients(self, data, model_len):
        
        grads = np.zeros(model_len)
        num_samples = len(data['y'])
        processed_samples = 0

        if num_samples < 50:
            input_data = process_x(data['x'])
            target_data = process_y(data['y'])
            with self.graph.as_default():
                model_grads = self.sess.run(self.grads, 
                    feed_dict={self.features: input_data, self.labels: target_data})
                grads = process_grad(model_grads)
            processed_samples = num_samples

        else:  # calculate the grads in a batch size of 50
            for i in range(min(int(num_samples / 50), 4)):
                input_data = process_x(data['x'][50*i:50*(i+1)])
                target_data = process_y(data['y'][50*i:50*(i+1)])
                with self.graph.as_default():
                    model_grads = self.sess.run(self.grads,
                    feed_dict={self.features: input_data, self.labels: target_data})

                flat_grad = process_grad(model_grads)
                grads = np.add(grads, flat_grad) # this is the average in this batch

            grads = grads * 1.0 / min(int(num_samples/50), 4)
            processed_samples = min(int(num_samples / 50), 4) * 50

        return processed_samples, grads
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
        '''
        x_vecs = process_x(data['x'], self.seq_len)
        labels = process_y(data['y'])
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()
