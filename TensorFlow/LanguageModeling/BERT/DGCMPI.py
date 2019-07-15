from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

class DGC_MPI_LIB:
   def init(self):
       print('Init in DGC_MPI')


   def decoder(tensor):
       flat_tensor = []
       for index in range(len(tensor)):
           if type(flat_grad[index]) is tuple:
               #expand the tuple
               count = flat_grad[index][0]
               #print(count)
               while count > 0:
                   flat_tensor.append(0)
                   count -= 1
           else:
               flat_tensor.append(flat_grad[index])
       return flat_tensor


   def _allreduce():
       size = comm.size
       if rank == 0:
           for rank in range(1, size):
               tensor = comm.recv(source=rank)
               flat_tensor += decoder(tensor)
           self.comm.Allreduce(flat_tensor, total_tensor, globalg, op=MPI.SUM)
           comm.bcast(total_tensor)
       else:
           comm.send(tensor,dest=0)





