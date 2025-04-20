'''Helper file for MPI initialization of different trajectories
   to different compute nodes for mean-field Ehrenfest simulation    '''
import HTightBinding as model
import sys
import os
sys.path.append('.')

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TrajDir = sys.argv[1]
NTraj = model.NTraj
NTasks = NTraj//size
NRem = NTraj - (NTasks*size)
TaskArray = [i for i in range(rank * NTasks , (rank+1) * NTasks)]
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)

for iTraj in TaskArray:
    os.system(f"python Ehrenfest_FFT.py {TrajDir} {iTraj}")