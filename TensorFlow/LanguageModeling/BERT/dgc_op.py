
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class CustomOptimizer(tf.train.Optimizer):

    def __init__(self, optimizer, name=None, use_locking=False):
        if name is None:
            name = "Distributed{}".format(type(optimizer).__name__)
        self._optimizer = optimizer
        self._stored_grads = []
        super(CustomOptimizer, self).__init__(name=name, use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
            """Compute gradients of all trainable variables.
            See Optimizer.compute_gradients() for more info.
            In DistributedOptimizer, compute_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)


