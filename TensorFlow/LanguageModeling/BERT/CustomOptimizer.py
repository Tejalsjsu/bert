import tensorflow as tf

class CustomOptimizer(tf.train.Optimizer):
   def __init__(self, optimizer, name=None):
       if name is None:
           name = "Custom{}".format(type(optimizer).__name__)

       self._optimizer = optimizer
   
   def compute_gradients(self, *args, **kwargs):
       grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)	
       # grads_and_vars = [(g,v) for g,v in grads_and_vars if g is not None]
	
       # sparsification starts here
       # calculating mean of gradients
       # threshold  = tf.reduce_mean(grads_and_vars[0])

       # using median instead of mean
       high = tf.contrib.distributions.percentile(grads_and_vars[0],50.,interpolation='higher')
       low = tf.contrib.distributions.percentile(grads_and_vars[0],50.,interpolation='lower')
       threshold = (high+low)/2

       # with tf.Session() as sess: sess.run(threshold)

       for grad, var in grads_and_vars:
           # get prev grad if it is in slot ans add it to current grad
           prev_grad  = self._optimizer.get_slot(var, 'prev_grad')
           # grad = tf.math.add(grad, prev_grad)
           if prev_grad is  None:
               continue
           else:
               grad += prev_grad

            
           # backed up grad that are less than threshold to use in next iteration
           bool_mask_less = tf.math.less(grad, threshold)
           float_mask_less = tf.cast(bool_mask_less, grad.dtype)
           backup_grads = tf.multiply(grad, float_mask)
           prev_grad  = self._optimizer.get_slot(var, 'prev_grad')

           # prev_grad  = optimizer._get_or_make_slot(var, var.initialized_value(), 'prev_grad', 'AdamWeightDecayOptimizer')
           # backup = optimizer._get_or_make_slot_with_initializer(var, var.initialized_value(), var.get_shape(), grad.dtype,  'backup_grads', 'AdamWeightDecayOptimizer')
           bool_mask = tf.math.greater(grad, threshold)
           float_mask = tf.cast(bool_mask, grad.dtype)
           grad = tf.multiply(grad, float_mask)
	
       # sparsification ends here
       return grads_and_vars


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
