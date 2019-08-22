from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import flearn.utils.tf_utils as tf_utils


class PROXSVRG(optimizer.Optimizer):
    """Implementation of Proximal SVRG, i.e., FedProx optimizer"""

    def __init__(self, learning_rate=0.001, lamb=0.001, use_locking=False, name="PROXSVRG"):
        super(PROXSVRG, self).__init__(use_locking, name)
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
            self._zeros_slot(v, "wzero", self._name)
            self._zeros_slot(v, "f_w_0", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        lamb_t = math_ops.cast(self._lamb_t, var.dtype.base_dtype)

        f_w_0 = self.get_slot(var, "f_w_0")
        vzero = self.get_slot(var, "vzero")
        wzero = self.get_slot(var, "wzero")
        v_n_s = grad - f_w_0 + vzero
        v_t = var - lr_t * v_n_s
        prox = tf_utils.prox_L2(v_t, wzero, lr_t, lamb_t)
        var_update = state_ops.assign(var, prox)

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

    def set_wzero(self, wzero, client):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, wzero):
                v = self.get_slot(variable, "wzero")
                v.load(value, client.sess)
