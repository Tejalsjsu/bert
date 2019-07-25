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

def _indexedslice(grad, shape):
    flat_grad = tf.reshape(grad, [-1])
    values, indices = [],[]
    row, col = 0, 0
    sess = tf.Session()
    
    for tensor in sess.run(flat_grad):
        if tensor != 0:
            values.append(tensor)
            indices.append((row , col))
        if col >= shape[1]:
            col = 0
            row += 1
        else:
            col += 1
        
    
    #print(values)
    #print(indices)
    
    indexed_sclices = tf.IndexedSlices(values,indices, dense_shape=x.shape)
    return indexed_sclices



class CustomOptimizer(tf.train.Optimizer):
   def __init__(self, optimizer, name=None):
       if name is None:
           name = "Custom{}".format(type(optimizer).__name__)
       self._name = name
       self._optimizer = optimizer
   
   def compute_gradients(self, *args, **kwargs):
       grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)	
       grads_and_vars = [(g,v) for g,v in grads_and_vars if g is not None]
       newgrads = []
       threshold = tf.contrib.distributions.percentile(abs(grads_and_vars[0][0]),80.0,interpolation='higher')
       for grad, var in grads_and_vars:
           #threshold = tf.contrib.distributions.percentile(grad,50.0,interpolation='higher')
           prev_grad = self._optimizer._get_or_make_slot(var, tf.zeros(tf.shape(grad), grad.dtype, 'prev_grad'), 'prev_grad', self._name)
           grad = tf.math.add(grad, prev_grad)
           #if prev_grad is  None:
           #    continue
           #else:
           #    grad += prev_grad
          
           # backed up grad that are less than threshold to use in next iteration
           bool_mask_less = tf.math.less(abs(grad), threshold)
           float_mask_less = tf.cast(bool_mask_less, grad.dtype)
           backup_grads = tf.multiply(grad, float_mask_less)
           #prev_grad  = self._optimizer.get_slot(var, 'prev_grad')
           prev_grad = self._optimizer._get_or_make_slot(var, backup_grads, 'prev_grad', self._name)
           bool_mask = tf.math.greater(abs(grad), threshold)
           float_mask = tf.cast(bool_mask, grad.dtype)
           #grad = tf.multiply(grad, float_mask)
           ''' 
           flat_grad = tf.reshape(grad, [-1])
           print('after flat')
           values, indices = [],[]
         
           row, col = 0, 0
           #tf.enable_eager_execution() 
           for tensor in sess.run(flat_grad):
               if tensor != 0:
                   values.append(tensor)
                   indices.append((row , col))
               if col >= shape[1]:
                   col = 0
                   row += 1
               else:
                   col += 1
        
          '''

           #indexed_sclices = tf.IndexedSlices(grad,grad,dense_shape=grad.shape)
           #newgrads.append(indexed_sclices)
           newgrads.append(tf.multiply(grad, float_mask))
           #Call indexSparsification over here

       return [(grad, gradvar[1]) for grad, gradvar in zip(newgrads, grads_and_vars)]

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
  

