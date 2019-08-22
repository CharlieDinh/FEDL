from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import flearn.utils.tf_utils as tf_utils


class SVRG(optimizer.Optimizer):
    """Implementation of SVRG optimizer"""

    def __init__(self, learning_rate=0.001, use_locking=False, name="SVRG"):
        super(SVRG, self).__init__(use_locking, name)
        self._lr = learning_rate
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            self._zeros_slot(v, "vzero", self._name)
            self._zeros_slot(v, "f_w_0", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        #mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        f_w_0 = self.get_slot(var, "f_w_0")
        vzero = self.get_slot(var, "vzero")
        v_n_s = grad - f_w_0 + vzero
        var_update = state_ops.assign_sub(var, lr_t * v_n_s)

        return control_flow_ops.group(*[var_update, ])

    def set_vzero(self, vzero, client):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, vzero):
                v = self.get_slot(variable, "vzero")
                v.load(value, client.sess)

    def set_fwzero(self, fwzero, client):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, fwzero):
                v = self.get_slot(variable, "f_w_0")
                v.load(value, client.sess)
