from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import flearn.utils.tf_utils as tf_utils
import numpy


class PROXSARAH(optimizer.Optimizer):
    """Implementation of Proximal Sarah, i.e., FedProx optimizer"""

    def __init__(self, learning_rate=0.001, lamb=0.001, use_locking=False, name="PROXSARAH"):
        super(PROXSARAH, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._lamb = lamb
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._lamb_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._lamb_t = ops.convert_to_tensor(self._lamb, name="lamb")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            self._zeros_slot(v, "vzero", self._name)
            self._zeros_slot(v, "preG", self._name)
            self._zeros_slot(v, "wzero", self._name)
            self._zeros_slot(v, "temp", self._name)

    # def _apply_dense(self, grad, var):
    #    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    #    lamb_t = math_ops.cast(self._lamb_t, var.dtype.base_dtype)

    #    vzero = self.get_slot(var, "vzero")
    #    preG = self.get_slot(var, "preG")
    #    wzero = self.get_slot(var, "wzero")

    #    v_zero_mean = tf.reduce_mean(vzero)
    #    print_op = tf.print('vzero: ', vzero)
     #   print_op_mean = tf.print('vzero mean: ', v_zero_mean)

    #    v_n_s = grad - preG + vzero
    #    print_vns = tf.print('vns:', v_n_s)

    #    with tf.control_dependencies([print_op, print_vns, print_op_mean]):
    #        v_update = state_ops.assign(vzero, v_n_s)
    #    v_t = var - lr_t * v_n_s
        #prox = tf_utils.prox_L2(var - lr_t * v_n_s, lamb_t)
    #    prox = tf_utils.prox_L2(v_t, wzero, lr_t, lamb_t)
    #    var_update = state_ops.assign(var, prox)

    #    return control_flow_ops.group(*[var_update, v_update, ])

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        lamb_t = math_ops.cast(self._lamb_t, var.dtype.base_dtype)

        vzero = self.get_slot(var, "vzero")
        preG = self.get_slot(var, "preG")
        wzero = self.get_slot(var, "wzero")

        v_n_s = grad - preG + vzero
        v_update = state_ops.assign(vzero, v_n_s)
        v_t = var - lr_t * v_n_s
        #prox = tf_utils.prox_L2(var - lr_t * v_n_s, lamb_t)
        prox = tf_utils.prox_L2(v_t, wzero, lr_t, lamb_t)
        var_update = state_ops.assign(var, prox)

        return control_flow_ops.group(*[var_update, v_update, ])

    def set_vzero(self, vzero, client):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, vzero):
                v = self.get_slot(variable, "vzero")
                v.load(value, client.sess)

    def set_preG(self, fwzero, client):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, fwzero):
                v = self.get_slot(variable, "preG")
                v.load(value, client.sess)

    def set_wzero(self, wzero, client):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, wzero):
                v = self.get_slot(variable, "wzero")
                v.load(value, client.sess)

    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        lamb_t = math_ops.cast(self._lamb_t, var.dtype.base_dtype)

        vzero = self.get_slot(var, "vzero")
        preG = self.get_slot(var, "preG")
        wzero = self.get_slot(var, "wzero")

        v_n_s = self.get_slot(var, "temp")

        # v_n_s = grad - preG + vzero
        temp = state_ops.assign(v_n_s, grad.values)
        with ops.control_dependencies([temp]):
            vns_update = state_ops.scatter_add(
                v_n_s, grad.indices, vzero - preG)
        with ops.control_dependencies([vns_update]):
            v_update = state_ops.assign(vzero, temp)
        v_t = var - lr_t * temp
        #prox = tf_utils.prox_L2(var - lr_t * v_n_s, lamb_t)
        prox = tf_utils.prox_L2(v_t, wzero, lr_t, lamb_t)
        var_update = state_ops.assign(var, prox)

        return control_flow_ops.group(*[var_update, v_update, ])
