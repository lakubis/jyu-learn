"""
mpi4py calculation of eigenvalues of multiple matrices

"""



from mpi4py import MPI

import matplotlib.pyplot as plt

import os
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS


import numpy as np
import scipy.linalg as linalg


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

def gen_data(N,shape):
    np.random.seed(123123123) # note
    mats = []
    for _ in range(N):
        m = np.random.random(shape)        
        mats.append(m@m.T)
    mats = np.array(mats)
    print(f'generated {N} random symmetric matrices, shape {shape}')
    return mats

def task(mat):
    e_vals, e_vecs  = linalg.eig(mat) # discard eigenvectors
    res = np.array(np.real(e_vals))   # keep result contiguous
    return res

if __name__=='__main__':
    N = 1000
    n = 300
    shape=(n,n)
    tag_end=1111 # arbitrary end signal
    

    if rank== 0:
        data = gen_data(N,shape)
        print(f'using {nproc} processes')
        assert nproc>1  # code runs only for more than one process
        tic = MPI.Wtime()
        dest = 1        
        for i in range(N):
            #print('dest',dest)
            comm.Send(data[i], dest=dest, tag=dest)
            dest = dest + 1
            if dest==nproc: dest=1
        for dest in range(1,nproc):
            comm.send(None, dest=dest, tag=tag_end)
        print(f'rank {rank} sending done')
    else:
        s = MPI.Status()
        while True:
            data= np.empty(shape,dtype=float)
            comm.Recv(data, source=0, tag=rank)
            # compute eigenvalues
            e_vals = task(data)
            # send result to rank 0            
            comm.Send(e_vals, dest=0)
            # probe if all is done 
            comm.probe(source=0,status=s)
            if s.tag==tag_end: break
            

    if rank==0:
        emax=[]
        e_vals= np.empty(n,dtype=float)
        for _ in range(N):
            comm.Recv(e_vals, source=MPI.ANY_SOURCE)
            #e_vals = comm.recv(source=MPI.ANY_SOURCE)
            emax.append(e_vals.max())

        print(f'max eigval {max(emax)}')
        toc = MPI.Wtime()
        print(f'calculations took {toc-tic} seconds')
        with open('timings.dat','a') as f:
            f.write(f'{nproc:<5d}  {toc-tic:<10.6f}\n')
        

    print(f'rank {rank} has finished')


    