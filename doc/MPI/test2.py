import numpy
import sys
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def MPI_partition(sendbuf,nprocs,colum):
    ##each nprocess total number: the size of each sub-task
    ave, res = divmod(sendbuf.size, nprocs*colum)
    ave1, res1 = divmod(res, colum)
    each_nprocess_row = np.array([ave + 1 if p < ave1 else ave  for p in range(nprocs)])
    total_number = each_nprocess_row*colum

    ##each nprocess star index: the starting index of each sub-task
    star_index = np.array([sum(total_number[:p]) for p in range(nprocs)])
    return each_nprocess_row,total_number,star_index


if __name__ == '__main__':
    if  rank == 0:
        file_list = os.listdir('D:\Desktop\MPI-document')
        sys.stderr.write("%d files\n" % len(file_list))
    else:
        file_list = None
        
    file_list = comm.bcast(file_list,root=0)
    
    num_files = len(file_list)
    