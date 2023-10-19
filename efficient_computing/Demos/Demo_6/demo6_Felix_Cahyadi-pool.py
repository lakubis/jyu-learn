"""
Assignments by: Felix Cahyadi
Date created: 16.10.2023
Last modified: -
"""

"""
Here, I'm going to implement the multiprocessing pool. The original version is demo6_Felix_Cahyadi.py

This is the output of the program for 100 matrices and Pool(4):

generated 100 random symmetric matrices, shape (300, 300)
max eigval 22797.220103452557
Total time = 2.459024429321289 s
PS E:\RADMEP_stuff\Semester 1 - Jyväskylä\jyu-learn> 

I tried using Pool(8) and Pool(12), but it doesn't increase the computation speed, rather it increases them. In case of Pool(12), it takes about 3 second to finish all of them.

But if I put 1000 matrices with Pool(4), it takes 16.158901929855347 s while 1000 matrices with Pool(8) takes 11.616026639938354 s.

The conclusion that I get is that the computation is only worth it to be parallelized if we have a lot of items. Maybe it takes a while to divide the task.

"""


"""
mpi4py calculation of eigenvalues of multiple matrices

"""



from mpi4py import MPI
from multiprocessing import Pool
from time import time as T

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
    N = 100
    n = 300
    shape=(n,n)

    data = gen_data(N, shape)
    
    tic = T()
    emax=[]
    with Pool(8) as p:
        evals = p.map(task, data) # We generate evals
        emax.append(np.max(evals)) # Append the maximum evals to emax
    toc = T()


    print(f'max eigval {np.max(emax)}') # get the maximum emax
    print(f"Total time = {toc-tic} s")
    



    