from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import flearn.utils.tf_utils as tf_utils
import numpy


class FEDL(optimizer.Optimizer):
    """Implementation of Proximal Sarah, i.e., FedProx optimizer"""

    def __init__(self, learning_rate=0.001,hyper_learning_rate = 0.001, lamb=0.001, use_locking=False, name="FEDL"):
        super(FEDL, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._hp_lr = hyper_learning_rate
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._hp_lr_t = ops.convert_to_tensor(self._hp_lr, name="hyper_learning_rate")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            self._zeros_slot(v, "preG", self._name)
            self._zeros_slot(v, "preGn", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        hp_lr_t = math_ops.cast(self._hp_lr_t, var.dtype.base_dtype)
        preG = self.get_slot(var, "preG")
        preGn = self.get_slot(var, "preGn")
        var_update = state_ops.assign_sub(var, lr_t*(grad + hp_lr_t*preG - preGn))
        #var_update = state_ops.assign_sub(var, w)

        return control_flow_ops.group(*[var_update,])

    def set_preG(self, preG, client):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, preG):
                v = self.get_slot(variable, "preG")
                v.load(value, client.sess)

    def set_preGn(self, preGn, client):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, preGn):
                v = self.get_slot(variable, "preGn")
                v.load(value, client.sess)

