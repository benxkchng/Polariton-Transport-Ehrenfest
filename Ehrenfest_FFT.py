import numpy as np
from numba import jit
import HTightBinding as model
import sys
import time

# Get path information and trajectory information
TrajFolder = sys.argv[1]
iTraj = int(sys.argv[2])
print(f"Running trajectory {iTraj}")

# Loads initial conditions
n0 = np.loadtxt('initCond.txt', dtype=np.complex128)
cFinit = n0

# Parameters for FFT of ψₘ which is equivalent to Hₗₘ @ ψₘ
nmol_indices = np.linspace(0, model.NMol-1, model.NMol)
common_factor = 1 / model.NMol
exp_fact_M = np.exp(1j * 2 * np.pi * common_factor * nmol_indices)

# parameters for FFT of ψₗ
pad_vecL = np.zeros(model.NMol - model.NMod, dtype=np.complex128)
phi_1 = np.exp(-1j * np.pi * np.linspace(0, model.NMol-1, model.NMol) / model.NMol)
phi_2 = np.exp(1j * np.pi * np.linspace(0, model.NMol-1, model.NMol) * model.NMod / model.NMol)

# Initialization of the mapping Variables
@jit(nopython=True)
def initMapping():
    qF = (np.real(cFinit)*np.sqrt(2)).astype(np.complex_)
    pF = (np.imag(cFinit)*np.sqrt(2)).astype(np.complex_)
    return qF, pF 

@jit(nopython=True)
def Hel_LM_psiM(vecM):
    """FFT of ψₘ which is equivalent to Hₗₘ @ ψₘ"""
    vecM_mod = vecM * exp_fact_M
    vecM_fft = np.fft.fftshift(np.fft.ifft(vecM_mod))[model.NMol//2 - (model.NMod//2 + 1) : model.NMol//2 + model.NMod//2]
    return vecM_fft * model.gc * model.NMol

@jit(nopython=True)
def Hel_ML_psiL(vecL):
    """FFT of ψₗ which is equivalent to Hₘₗ @ ψₗ"""
    vecL_mod = vecL * model.gc
    vecL_pad = np.hstack((vecL_mod, pad_vecL))
    vecL_fft = np.fft.fft(vecL_pad)
    return vecL_fft * phi_1 * phi_2
    

@jit(nopython=True)
def htcMatMul(H, vec):
    """ Hardcoding multiplication Hamiltonian matrix to a vector """
    vecF = np.zeros((len(vec)), dtype=np.complex128)
    
    vecM = vec[1: model.NMol + 1]       # matter wavefunction
    vecL = vec[model.NMol + 1 : ]       # light wavefunction
    
    # Diagonal element of Hel
    vecF = vecF + (H * vec)#(H_diag * vec)
        
    # Light matter interation 
    vecF[1: model.NMol + 1] = vecF[1: model.NMol + 1] + Hel_ML_psiL(vecL)
    vecF[model.NMol + 1 : ] = vecF[model.NMol + 1 : ] + Hel_LM_psiM(vecM)    

    return vecF

@jit(nopython=True)
def Umap(qF, pF, dt, H):
    """ Unitary step """
    qFin, pFin = qF*1.0, pF*1.0 # Store input position and momentum for verlet propogation

    # RK4 propagation    
    c = (qFin+1j*pFin)/np.sqrt(2)
        
    ck1 = (-1j) * htcMatMul(H,c)  #(-1j) * (VMat @ c)
    ck2 = (-1j) * htcMatMul(H,(c + (dt/2.0) * ck1 )) #(-1j) * (VMat @ (c + (dt/2.0) * ck1 ))
    ck3 = (-1j) * htcMatMul(H,(c + (dt/2.0) * ck2 ))#(-1j) * (VMat @ (c + (dt/2.0) * ck2 ))
    ck4 = (-1j) * htcMatMul(H, (c + (dt) * ck3 )) #(-1j) * (VMat @ (c + (dt) * ck3 ))
    c = c + (dt/6.0) * (ck1 + 2.0 * ck2 + 2.0 * ck3 + ck4)
    
    qF = (np.real(c)*np.sqrt(2)).astype(np.complex_)
    pF = (np.imag(c)*np.sqrt(2)).astype(np.complex_)

    return qF, pF

@jit(nopython=True)
def Force(R, qF, pF):
    """ Hard-code force terms directly from parameters as in Orlando and Troissi """
    dH0 = model.dHel0(R)
    F = np.zeros((len(R)))
    F = -dH0
    
    cF = (qF + 1j*pF)/np.sqrt(2)  
    
    for j in range(model.NMol):
        F[j * model.NModes : (j + 1) * model.NModes] -=  model.ck[:] * ( np.abs(cF[j+1]) ** 2 )
    
    return F

@jit(nopython=True)
def VelVer(R, P, qF, pF, dtI, dtE, F1,  M):
    """ Ehrenfest propagation with velocity verlet """
    EStep = int(dtI/dtE)
    v = P/M
    
    H_diag = model.Hel_diag(R)
    for t in range(EStep):
        qF, pF = Umap(qF, pF, dtE/2, H_diag) # Unitary mapping
    R += v * dtI + 0.5 * F1 * dtI ** 2 / M   # Update position R
    H_diag = model.Hel_diag(R)
    for t in range(EStep):
        qF, pF = Umap(qF, pF, dtE/2, H_diag) # Unitary mapping
    
    F2 = Force(R, qF, pF)                    # Update force
    v += 0.5 * (F1 + F2) * dtI / M           # Update momentum P
    
    return R, v*M, qF, pF, F2

@jit(nopython=True)
def pop(qF, pF):
    return np.outer((qF + 1j*pF), qF-1j*pF)/2

@jit(nopython=True)
def coeffs(qF, pF):
    c = qF - (1j * pF)          # ρ = np.outer(np.conj(c), c)/2
    return c.real, c.imag       # c = c.real + 1j*c.imag

@jit(nopython=True)
def runTraj(iR, iP):

    ## Parameters -------------
    dtE = model.dtE
    dtN = model.dtN
    NSteps = model.NSteps
    NStates = model.NStates
    M = model.M # mass

    nskip = model.nskip

    #---------------------------
    
    # # Trajectory data
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
                
        
    rho_ensemble = np.zeros((NSteps//nskip + pl, 2*NStates+1))                    # 2 is to store real and imaginary component seperately
    rho_ensemble[:,0] = np.arange(0,(NSteps//nskip + pl)*model.dtN,model.dtN)

    Ravg = np.zeros((NSteps//nskip))
    
    # Trajectory data
    R, P = iR, iP
    qF, pF = initMapping() 

    #----- Initial Force --------
    F1 = Force(R, qF, pF)
    iskip = 0
    for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
        if (i % nskip == 0):

            cReal, cImag = coeffs(qF, pF)
            rho_ensemble[iskip, 1: 2*NStates+1] += np.hstack((cReal, cImag))
            
            Ravg[iskip] += R[0] 
            iskip += 1

        R, P, qF, pF, F1 = VelVer(R, P, qF, pF, dtN, dtE, F1, M)
            
    return rho_ensemble


# Loads R, P from each trajectory
NuclearData = np.loadtxt(TrajFolder + f"{iTraj+1}/initial_bath_{iTraj+1}.txt")
iR, iP = NuclearData[:, 0], NuclearData[:, 1]

# Runs dynamics for one trajectory
dyn_st = time.time()
rho_lessMem = runTraj(iR, iP)
dyn_end = time.time()
print(f"It took {dyn_end-dyn_st} seconds to complete the dynamics")

# Saves coefficients to disk for one trajectory
save_st = time.time()
np.savetxt(TrajFolder + f"{iTraj+1}/Pij_FFT_{iTraj+1}.txt", rho_lessMem, fmt='% 14.8e')
save_end = time.time()
print(f"It took {save_end-save_st} seconds to save the data")
