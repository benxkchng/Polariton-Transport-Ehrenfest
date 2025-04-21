import numpy as np
from numba import jit

@jit(nopython=True)
def bathParam(lamba, ωc,M, ndof):
    """ Calculates phonon coupling strength and frequency for Drude bath """
    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))
    for d in range(ndof):
        # Debye bath
        ω[d] = ωc * np.tan((np.pi/2)*(1 - (d+1)/(ndof+1)  ))
        c[d] = np.sqrt(2*M*lamba/(ndof+1)) * ω[d]  
    return c, ω

# List of conversion constants to au
fs2au = 41.341374575751
cminv2au = 4.55633*1e-6
eV2au = 0.036749405469679
K2au = 0.00000316678
amu = 1822.888
ps = 41341.374575751
ang = 1.8897
au = 27.2114
c = 137.0 

'''dtN -> nuclear time step (au)
    Nsteps -> number of nuclear time steps
    Total simulation time = Nsteps x dtN (au)'''
SimTime = 200             # in fs
dtN = 100.0               # in au originally 40
NSteps = int(SimTime/(dtN/fs2au)) + 1  

'''Esteps -> number of electronic time steps per nuclear time step
    dtE -> electronic time step (au)'''    
ESteps = 500                
dtE = dtN/ESteps     

NMol = 20001                              # Number of molecules
NMod = 281                                # Number of cavity modes
NStates = NMol + NMod + 1                 # number of electronic states
M = 1                                     # mass of nuclear particles (au)
NTraj = 500                               # number of trajectories
nskip = 1                                 # save data every nskip steps of PLDM simulation

'''Bath parameters'''
NModes = 35                               # Number of bath modes per sites
lambda_ = 0.006/au                        # Reorganization energy
ωc = 0.0061992128657421585/au             # Characteristic frequency
ck, ωk = bathParam(lambda_, ωc,M, NModes)
lambda_bath = 0.5*np.sum(ck**2/(M*ωk**2)) # Consistency check for bath parameters
NR = NMol * NModes

eExc = 1.96 /au                           # Exciton energy
gc0 =  0.120 / au / np.sqrt(NMol)         # Light-matter coupling strength

Lx = 40*ang                               # Lattice spacing
wc0 = 1.90/au                             # Cavity fundamental frequency
kz = wc0/c
kx = 2 * np.pi *  np.arange(-NMod//2 + 1,NMod//2+1) / (NMol*Lx)
kx = kx[:NMod] 
nr = 1.0                                  # refractive index
ωc = (c/nr) * (kx**2.0 + kz**2.0)**0.5

# Calculate gc0 for TM polarization
θ = np.arctan(kx/kz)
gc = gc0 * (ωc/wc0)**0.5  * np.cos(θ)
xj = np.arange(0,NMol) * (Lx)             # integer positions

@jit(nopython=True)
def Hel_diag(R):
    """ Constructs diagonal component of HTC matrix """
    Vii = np.zeros(NStates, dtype=np.complex128)
    for iState in range(NMol):
        Vii[iState] = eExc + lambda_ + np.sum(ck[:]*R[(iState)*NModes:(iState+1)*NModes])
    Vii[NMol+1: ] = ωc
    return Vii

@jit(nopython=True)
def dHel0(R):
    '''Bath derivative of the state independent part of the Hamiltonian'''
    dH0 = np.zeros((len(R)))
    for j in range(NMol):
        dH0[j * NModes : (j + 1) * NModes] = ωk**2 * R[j * NModes : (j + 1) * NModes] 

    return dH0
