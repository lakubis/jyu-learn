"""
Assignments by: Felix Cahyadi
Date created: 16.10.2023
Last modified: -
"""

"""
Here, I'm going to implement the multiprocessing pool. The original version is demo6_Felix_Cahyadi.py

"""


"""
mpi4py calculation of eigenvalues of multiple matrices

"""



from mpi4py import MPI
from multiprocessing import Pool

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
    
    


    