import tensorflow as tf
import re

from compression import Compression
from DGCMPI import DGC_MPI_LIB

def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)

def _compress(grad):
   shape = tf.shape(grad)
   flat = tf.reshape(grad, [-1])
   sess = tf.Session()
   #print(sess.run(flat), flat)  
   #print(sess.run(shape))
   compressed, zeroCnt = [], 0

   for item in sess.run(flat):
       if item == 0:
           zeroCnt += 1
       else:
           if zeroCnt > 0:
               cprd.append((zeroCnt, 0))
               zeroCnt = 0
           compressed.append(item)
   if zeroCnt > 0:
       compressed.append((zeroCnt, 0))
       zeroCnt = 0

   print(compressed)
   return compressed, shape

def _decompress(grad, shape):
   matrix, sub_list = [], []
   outer, inner = 0, 0
   flat_grad_index = 0
    
   while outer < shape[0]:
       while inner < shape[1]:
           #print(flat_grad[flat_grad_index])
           if type(flat_grad[flat_grad_index]) is tuple:
               #expand the tuple
               count = flat_grad[flat_grad_index][0]
               #print(count)
               while count > 0:
                   sub_list.append(0)
                   count -= 1
                   inner += 1
           else:
               sub_list.append(flat_grad[flat_grad_index])
               inner += 1
           flat_grad_index += 1
       matrix.append(sub_list)
       sub_list = []
       outer += 1
       inner = 0
    
   #print(matrix)
    
   #unflatten = tf.convert_to_tensor(matrix, dtype=tf.float32)
   unflatten = tf.constant(matrix)
   print(sess.run([unflatten]))
   return sess.run(unflatten)

def _allreduce(tensor, name=None):
    """An op which sums an input tensor over all the Horovod processes.
    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Horovod processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.
    Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.
    """
    if name is None:
        name = 'Allreduce_%s' % _normalize_name(tensor.name)
    # call All compress over here to compress gradients
    tensor, shape = _compress(tensor)
    tensor = DGC_MPI_LIB._allreduce(tensor, name=name)
    return _decompress(tesnor, shape)

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
       threshold = tf.contrib.distributions.percentile(grads_and_vars[0],10.,interpolation='higher')
       #low = tf.contrib.distributions.percentile(grads_and_vars[0],50.,interpolation='lower')
       #threshold = (high+low)/2

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

           dgc_mpi = DGC_MPI_LIB()
           grads_and_vars[0] = dgc_mpi._allreduce(grads_and_vars[0])
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
   
   def allreduce(tensor,average=True, device_dense='', device_sparse='', compression=Compression.none):
       """ Perform an allreduce on a tf.Tensor
        
        Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
                The shape of the input must be identical across all ranks.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.
        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was built with HOROVOD_GPU_ALLREDUCE.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was built with HOROVOD_GPU_ALLGATHER.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.

        Returns:

        A tensor of the same shape and type as `tensor`, summed across all
        processes.

        """
       horovod_size = tf.cast(size(), dtype=tensor.dtype)
       tensor_compressed, ctx = compression.compress(tensor)
       summed_tensor_compressed = _allreduce(tensor_compressed)
       summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
       new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

       return new_tensor
