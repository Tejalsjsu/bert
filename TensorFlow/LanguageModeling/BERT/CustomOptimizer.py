import tensorflow as tf

class CustomOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, name=None):
        if name is None:
            name = "Custom{}".format(type(optimizer).__name__)

        self._optimizer = optimizer
   
    def compute_gradients(self, *args, **kwargs):

        return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "prev_grad", self._name)
