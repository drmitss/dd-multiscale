# CSPmodule.py
import os
import numpy as np
import scipy as sp
import sympy as smp
from sympy import pprint
import pandas as pd
# from sympy import Matrix
from scipy.linalg import eig, inv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from joblib import Parallel, delayed, parallel_backend
import warnings
from tqdm import TqdmExperimentalWarning

# silence that “Using `tqdm.autonotebook.tqdm` in notebook mode” warning forever
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
from tqdm_joblib import tqdm_joblib
tqdm_joblib.tqdm = tqdm

GREEN = "\033[92m"
GRAY = "\033[90m"
RESET = "\033[0m"

bar_format = f"{{l_bar}}{GREEN}{{bar}}{RESET} {{elapsed}}"

import model_module as model
model.init_symbolic_model(model.nSpec,model.kReac)

##########################################################################################
def eval_RR(y, parameters):
    return np.array(model.RR(y, parameters))
def eval_gradR(y, parameters):
    return np.array(model.gradR(y, parameters))
def eval_g(y, parameters):
    return np.array(model.g(y, parameters))
def eval_jac(y, parameters):
    return np.array(model.jac(y, parameters))
def eval_jac_R(y, parameters):
    return np.array(model.jac_R(y, parameters))
##########################################################################################

def colors_fun(size):
    colors = ['red', 'blue', 'green', 'orange', 'black', 'purple', 'brown', 'pink', 'grey', 'navy', 'maroon', 'crimson', 'teal', 'salmon', 'turquoise', 'violet', 'indianred', 'lightgreen', 'darkkhaki', 'cyan', 'magenta', 'goldenrod', 'blueviolet', 'sandybrown', 'crimson', 'darksalmon', 'darkturquoise', 'darkviolet', 'brown', 'darkgreen', 'olive', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'indigo', 'saddlebrown', 'darkred']
    if size > len(colors): colors += colors * (size // len(colors))
    return colors

def colors_fun_reac(size):
    colors = ['salmon', 'turquoise', 'violet', 'indianred', 'darkgreen', 'crimson', 'lime', 'olive', 'cyan', 'chocolate', 'gold', 'indigo',  'magenta', 'teal', 'beige','black', 'purple', 'brown', 'pink', 'navy', 'maroon' ]
    if size > len(colors): colors += colors * (size // len(colors))
    return colors


def generate_log_ticks(data_min, data_max):
    # Determine the range of exponents
    min_exp = np.floor(np.log10(data_min))
    max_exp = np.ceil(np.log10(data_max))
    
    # Generate tick values and labels
    tickvals = [10**exp for exp in range(int(min_exp), int(max_exp) + 1)]
    ticktext = [f"$10^{{{int(exp)}}}$" for exp in range(int(min_exp), int(max_exp) + 1)]
    
    return tickvals, ticktext

##########################################################################################
def diagn(nspec, kreac, y, API_changed_sign=False):
    t_points = y[0].size
    Rates = np.zeros((kreac+1, t_points))           # Reaction Rates
    eigens = np.zeros((2*nspec+1, t_points))        # system's eigenvalues
    timescales = np.zeros((nspec+1, t_points))      # system's timescales
    epsilon = np.zeros(t_points)                    # perturbation value epsilon 
    exp_timescales = np.zeros((nspec+1, t_points))  # explosive timescales
    amplitudes = np.zeros((nspec+1, t_points))      # CSP Amplitudes
    ALPHAs = np.zeros((nspec+1, nspec+1, t_points)) # CSP alpha basis vectors
    BETTAs = np.zeros((nspec+1, nspec+1, t_points)) # CSP betta basis vectors
    APIs = np.zeros((nspec+1, kreac+1, t_points))   # CSP Amplitude Participation Index
    TPIs = np.zeros((nspec+1, kreac+1, t_points))   # CSP Timescale Participation Index
    Pointer = np.zeros((nspec+1,nspec+1, t_points)) # CSP Pointer (modal)
    IIs = np.zeros((nspec+1, kreac+1, t_points))    # CSP Importance Index
    
    tqdm._instances.clear()
    for t_point in tqdm(range(t_points), desc="♾️ CSP analysis", bar_format=bar_format, ascii='░█'):

    # Calculate Rates -----------------------------------------------------------------------
        RRs = model.eval_RR(y[1:,t_point], model.parameters(y[1:,t_point]))
        Rates[1:,t_point] = RRs
    
    # Calculate Jacobian --------------------------------------------------------------------
        jac = model.eval_jac(y[1:,t_point], model.parameters(y[1:,t_point]))
    
    # Calculate eigenvalues/eigenvectors ----------------------------------------------------
        evalSM, a_vec, b_vec, evalRI = eigen(nspec, jac)
        eigens[1:,t_point] = evalRI.flatten()
     
    # Calculate timescales (all/explossive) -------------------------------------------------
        timescales[1:,t_point], exp_timescales[1:,t_point] = calculate_timescales(evalRI)
        epsilon[t_point] = timescales[1, t_point] / timescales[2, t_point]
    # Calculate CSP Amplitudes, fi, API, and rescale a_vec.b_vec s.t. fi>0 ------------------
        # Calculates the contribution of the kth reaction to the generation of the nth amplitude f^n
        # A relatively large value of |API_k^n| indicates that the kth reaction 
        # provides a significant contribution to the generation of the nth amplitude f^n
        # while a relatively small value of |TPI_k^n| indicates a negligible contribution.
        API, Fi, BdS, a_vec, b_vec = calculate_API(nspec, kreac, y[1:,t_point], a_vec, b_vec, API_changed_sign)
        amplitudes[1:,t_point] = Fi
        APIs[1:,1:,t_point] = API
        ALPHAs[1:,1:,t_point] = a_vec
        BETTAs[1:,1:,t_point] = b_vec
    
    # Calculate CSP Pointer (modal) ---------------------------------------------------------
        # Po_n^m measures the relation of the nth component of y to the a_mf^m mode
        Po = calculate_Pointer(nspec, evalRI, a_vec, b_vec)
        Pointer[1:,1:,t_point] = Po
        
    # Calculate CSP TPI ---------------------------------------------------------------------
        # Calculates the contribution of the kth reaction to the generation of the nth eigenvalue (time scale)
        # A relatively large value of |TPI_k^n| indicates that the kth reaction 
        # provides a significant contribution to the generation of the nth time scale
        # while a relatively small value of |TPI_k^n| indicates a negligible contribution.
        TPI, BdS = calculate_TPI(nspec, kreac, y[1:,t_point], evalRI, a_vec, b_vec)
        TPIs[1:,1:,t_point] = TPI
    
    # Calculate CSP II ----------------------------------------------------------------------
        # Calculates the contribution of the kth reaction 
        # to the slow evolution of the nth element in y on the SIM created by NoEM
        # A relatively large value of |II_k^n| indicates that the kth reaction 
        # provides a significant contribution to the evolution of y^n on the SIM, 
        # while a relatively small value of |II_k^n| indicates a negligible contribution.
        NoEM = 2; # Set manually
        II = calculate_II(nspec, kreac, y[1:,t_point], a_vec, b_vec, NoEM).round(3)
        IIs[1:,1:,t_point] = II
    return Rates, ALPHAs, BETTAs, eigens, timescales, exp_timescales, amplitudes, TPIs, APIs, Pointer, IIs

##########################################################################################
##########################################################################################
def _process_timepoint(t_point, y, nspec, kreac, API_changed_sign):
    # exactly the body of your loop, but returning the per-t arrays instead of storing in a big pre-alloc
    Rates_t        = np.zeros(kreac+1)
    ALPHAs_t       = np.zeros((nspec+1, nspec+1))
    BETTAs_t       = np.zeros((nspec+1, nspec+1))
    eigens_t       = np.zeros(2*nspec+1)
    timescales_t   = np.zeros(nspec+1)
    exp_ts_t       = np.zeros(nspec+1)
    amplitudes_t   = np.zeros(nspec+1)
    TPIs_t         = np.zeros((nspec+1, kreac+1))
    APIs_t         = np.zeros((nspec+1, kreac+1))
    Pointer_t      = np.zeros((nspec+1, nspec+1))
    IIs_t          = np.zeros((nspec+1, kreac+1))
    epsilon_t      = 0.0

    # Rates
    RRs = model.eval_RR(y[1:,t_point], model.parameters(y[1:,t_point]))
    Rates_t[1:] = RRs

    # Jacobian & eigen
    jac = model.eval_jac(y[1:,t_point], model.parameters(y[1:,t_point]))
    evalSM, a_vec, b_vec, evalRI = eigen(nspec, jac)
    eigens_t[1:] = evalRI.flatten()

    # timescales
    timescales_t[1:], exp_ts_t[1:] = calculate_timescales(evalRI)
    epsilon_t = timescales_t[1] / timescales_t[2]

    # API, amplitudes, ALPHAs, BETTAs
    API, Fi, BdS, a_vec, b_vec = calculate_API(nspec, kreac, y[1:,t_point], a_vec, b_vec, API_changed_sign)
    amplitudes_t[1:] = Fi
    APIs_t[1:,1:]   = API
    ALPHAs_t[1:,1:] = a_vec
    BETTAs_t[1:,1:] = b_vec

    # Pointer
    Po = calculate_Pointer(nspec, evalRI, a_vec, b_vec)
    Pointer_t[1:,1:] = Po

    # TPI
    TPI, BdS = calculate_TPI(nspec, kreac, y[1:,t_point], evalRI, a_vec, b_vec)
    TPIs_t[1:,1:] = TPI

    # II
    NoEM = 2
    II = calculate_II(nspec, kreac, y[1:,t_point], a_vec, b_vec, NoEM).round(3)
    IIs_t[1:,1:] = II

    return (
        Rates_t, ALPHAs_t, BETTAs_t,
        eigens_t, timescales_t, exp_ts_t,
        amplitudes_t, TPIs_t, APIs_t,
        Pointer_t, IIs_t, epsilon_t
    )
##########################################################################################
def diagn_parallel(nspec, kreac, y, API_changed_sign=False, n_jobs=-1):
    T = y.shape[1]
    # create a single notebook‐style bar
    # # from tqdm import tqdm
    bar = tqdm_nb(total=T
               ,
               desc="♾️ CSP analysis",
               bar_format=f"{{l_bar}}{GREEN}{{bar}}{RESET} {{elapsed}}",
               ascii='░█')

    # with tqdm_joblib(bar), \
    #      parallel_backend('threading', n_jobs=n_jobs):
    #     results = Parallel(verbose=0)(
    #         delayed(_process_timepoint)(t, y, nspec, kreac, API_changed_sign)
    #         for t in range(T)
    #     )

    with tqdm_joblib(bar):
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_process_timepoint)(t, y, nspec, kreac, API_changed_sign)
            for t in range(T)
        )

    # # from tqdm.notebook import tqdm
    # bar = tqdm(
    #     total=T,
    #     desc="♾️ CSP analysis",
    #     colour="green",                 # fills green; empty remains grey
    #     leave=True
    # )
    # with tqdm_joblib(bar):
    #     results = Parallel(n_jobs=n_jobs, verbose=0)(
    #         delayed(_process_timepoint)(t, y, nspec, kreac, API_changed_sign)
    #         for t in range(T)
    #     )
    
    # Unpack results into big arrays
    Rates, ALPHAs, BETTAs, eigens, timescales, exp_ts, amplitudes, TPIs, APIs, Pointer, IIs, epsilons = zip(*results)

    Rates      = np.stack(Rates,      axis=1)
    ALPHAs     = np.stack(ALPHAs,     axis=2)
    BETTAs     = np.stack(BETTAs,     axis=2)
    eigens     = np.stack(eigens,     axis=1)
    timescales = np.stack(timescales, axis=1)
    exp_timescales = np.stack(exp_ts, axis=1)
    amplitudes    = np.stack(amplitudes, axis=1)
    TPIs          = np.stack(TPIs,     axis=2)
    APIs          = np.stack(APIs,     axis=2)
    Pointer       = np.stack(Pointer,  axis=2)
    IIs           = np.stack(IIs,      axis=2)
    epsilon       = np.array(epsilons)

    return (
        Rates, ALPHAs, BETTAs,
        eigens, timescales, exp_timescales,
        amplitudes, TPIs, APIs,
        Pointer, IIs
    )

##########################################################################################
##########################################################################################
##########################################################################################
#                                 DATA DRIVEN CSP                                        #
##########################################################################################
##########################################################################################
##########################################################################################
def _process_data_point(
    t_point, nspec,
    jacobian_estimate, fx, API_changed_sign
):
    """
    Worker for a single time‐point t_point.
    Returns: (timescales, exp_timescales, amplitudes, Pointer)
    """
    # 1) Jacobian & eigen
    evalSM, a_vec, b_vec, evalRI = eigen(nspec, jacobian_estimate[t_point])
    # print('1 Jacobian & eigen: passed')

    # 2) timescales
    timescales, exp_ts = calculate_timescales(evalRI)
    # print('2 timescales: passed')

    # 3) epsilon
    # eps = timescales[1] / timescales[2]

    # 4) amplitudes
    Fi, a_vec, b_vec = calculate_amplitudes_data(fx[t_point], a_vec, b_vec, API_changed_sign=API_changed_sign)
    # print('3-4 amplitudes: passed')

    # 5) Pointer
    Po = calculate_Pointer(nspec, evalRI, a_vec, b_vec)
    # print('5 Pointer: passed')

    return timescales, exp_ts, Fi, Po


def calculate_amplitudes_data(dydt, ALPHA, BETA, API_changed_sign=True):
    """
    Compute amplitude vector Fi = BETA @ dydt and optionally flip signs
    of Fi, ALPHA (columns), and BETA (rows) to enforce positive Fi.

    Parameters
    ----------
    dydt : ndarray, shape (nspec,)
        Time derivatives of state y.
    ALPHA : ndarray, shape (nspec, nspec)
        Right‐eigenvector matrix (columns are eigenvectors).
    BETA : ndarray, shape (nspec, nspec)
        Left‐eigenvector matrix (rows are eigenvectors).
    API_changed_sign : bool, optional
        If True, flip signs so that each Fi[i] >= 0. Default False.

    Returns
    -------
    Fi : ndarray, shape (nspec,)
        Amplitude vector BETA @ dydt (possibly sign‐flipped).
    ALPHA_flipped : ndarray, shape (nspec, nspec)
        ALPHA with columns flipped to match Fi signs.
    BETA_flipped : ndarray, shape (nspec, nspec)
        BETA with rows flipped to match Fi signs.
    """
    # Compute amplitude vector
    Fi = BETA @ dydt

    ALPHA_flipped = ALPHA.copy()
    BETA_flipped  = BETA.copy()

    if API_changed_sign:
        # Determine sign flips: enforce Fi[i] >= 0
        signFI1 = np.sign(Fi)
        signFI1[signFI1 == 0] = 1

        # Apply flips
        for i in range(Fi.size):
            ALPHA_flipped[:, i] *= signFI1[i]
            BETA_flipped[i, :]  *= signFI1[i]
            Fi[i]              *= signFI1[i]

    return Fi, ALPHA_flipped, BETA_flipped




def data_diagn_parallel(
    nspec,
    jacobian_estimate, fx,
    API_changed_sign=True,
    n_jobs=-1
):
    """
    Run the CSP‐diagnostics at each t in parallel.

    Returns:
      timescales_estimate      (ndarray (nspec+1, T))
      exp_timescales_estimate  (ndarray (nspec+1, T))
      amplitudes_estimate      (ndarray (nspec+1, T))
      Pointer_estimate         (ndarray (nspec+1, nspec+1, T))
    """
    T = fx.shape[0]

    # one single progress bar (console)
    bar = tqdm(total=T,
               desc="♾️ CSP analysis",
               bar_format=f"{{l_bar}}{GREEN}{{bar}}{RESET} {{elapsed}}",
               ascii='░█'
              )

    with tqdm_joblib(bar):
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_process_data_point)(
                t, nspec,
                jacobian_estimate, fx, API_changed_sign
            )
            for t in range(T)
        )

    # unzip
    ts_list, ets_list, amp_list, ptr_list = zip(*results)

    # stack into “raw” arrays of shape (nspec, T) or (nspec,nspec,T)
    ts_raw  = np.column_stack(ts_list)   # (nspec,    T)
    ets_raw = np.column_stack(ets_list)  # (nspec,    T)
    amp_raw = np.column_stack(amp_list)  # (nspec,    T)
    ptr_raw = np.stack(ptr_list, axis=2) # (nspec, nspec, T)

    # now pad them to (nspec+1,...)
    timescales_estimate     = np.zeros((nspec+1, T))
    exp_timescales_estimate = np.zeros((nspec+1, T))
    amplitudes_estimate     = np.zeros((nspec+1, T))
    Pointer_estimate        = np.zeros((nspec+1, nspec+1, T))

    # fill rows 1..nspec (and cols 1..nspec for Pointer)
    timescales_estimate   [1: , :]    = ts_raw
    exp_timescales_estimate[1: , :]   = ets_raw
    amplitudes_estimate   [1: , :]    = amp_raw
    Pointer_estimate      [1: , 1: , :] = ptr_raw

    return (
        timescales_estimate,
        exp_timescales_estimate,
        amplitudes_estimate,
        Pointer_estimate
    )
##########################################################################################
##########################################################################################
##########################################################################################
def eigen(nspec, Jac):

    # Compute eigenvalues and right eigenvectors
    eigvalues, eigVectors = sp.linalg.eig(Jac)
    d = eigvalues
    ind = np.argsort(-np.abs(d))

    evalSM = np.abs(d[ind])
    evalRI = np.zeros((nspec, 2))  # real, imaginary parts
    evalRI[:, 0] = np.real(d[ind])
    evalRI[:, 1] = np.imag(d[ind])

    Vs = eigVectors[:, ind]  # Sorted eigenvectors
    
    ALPHA = np.zeros((nspec, nspec))
    countIm = []
    for i in range(nspec):
        if np.isreal(d[ind][i]):
            ALPHA[:, i] = np.real(Vs[:, i])
        else:
            countIm.append(i)
            if i < nspec - 1:
                if len(countIm) % 2 == 1:
                    ALPHA[:, i] = np.real(Vs[:, i])
                    ALPHA[:, i + 1] = -np.imag(Vs[:, i])
    
    Ws = inv(Vs)
    BETA = np.zeros((nspec, nspec))
    countIm1 = []
    for i in range(nspec):
        if np.isreal(d[ind][i]):
            BETA[i, :] = np.real(Ws[i, :])
        else:
            countIm1.append(i)
            if i < nspec - 1:
                if len(countIm1) % 2 == 1:
                    BETA[i, :] = 2 * np.real(Ws[i, :])
                    BETA[i + 1, :] = 2 * np.imag(Ws[i, :])
    
    # Normalize ALPHA and BETA
    indM = np.argmax(np.abs(ALPHA), axis=0)
    for i in range(nspec):
        normFac = ALPHA[indM[i], i]
        if i in countIm and (i + 1) in countIm:
            compV = np.sqrt(ALPHA[:, i]**2 + ALPHA[:, i + 1]**2)
            ind = np.argmax(compV)
            c1 = 1 / compV[ind]**2
            c2, c3 = ALPHA[ind, i], ALPHA[ind, i + 1]
            normMat = np.array([[c1*c2, -c1*c3], [c1*c3, c1*c2]])
            ALPHA[:, i:i+2] = ALPHA[:, i:i+2] @ normMat
            BETA[i:i+2, :] = inv(normMat) @ BETA[i:i+2, :]
        else:
            ALPHA[:, i] = ALPHA[:, i] / normFac
            BETA[i, :] = BETA[i, :] * normFac

    return evalSM, ALPHA, BETA, evalRI


##########################################################################################
def calculate_timescales(evalRI):
    threshold = 1e-18
    timescales = np.zeros(evalRI.shape[0])
    exp_timescales = np.zeros(evalRI.shape[0])
    # Initialize exptimescales as an empty array; we'll append to this conditionally
    # exp_timescales = []
    
    for i in range(evalRI.shape[0]):
        # timescale = 1 / (np.abs(evalRI[:,0]+1j*evalRI[:,1]) + threshold)
        timescale = 1 / (np.linalg.norm(evalRI[i, :], 2) + threshold)
        timescales[i] = timescale
        
        # If the first element of the row is positive, handle the exptimescales logic
        if evalRI[i, 0] > 0:
            # Append the timescale to exptimescales
            # exp_timescales.append(timescale)
            exp_timescales[i] = timescale
    # exp_timescales = np.array(exp_timescales)
    return timescales, exp_timescales



##########################################################################################
def calculate_API(nspec, kreac, yy, ALPHA, BETA, API_changed_sign):
    ALPHAs = ALPHA
    BETAs = BETA
    SS = model.stoicVect(nspec, kreac)
    RR = eval_RR(yy, model.parameters(yy))

    BdS = BETA @ SS  # Matrix multiplication in Python for BETA*SS
    
    SRk = np.zeros((nspec, kreac))
    for i in range(kreac):
        SRk[:, i] = SS[:, i] * RR[i]  # S^i_k * R^k
    
    BSRk = BETA @ SRk
    FI1 = np.sum(BSRk, axis=1)
    Fi = FI1
    FI2 = np.sum(np.abs(BSRk), axis=1) + 1e-18
    normAPI = 1. / FI2
    API = np.zeros((nspec, kreac))
    signFI1 = np.sign(FI1)
    signFI1[signFI1 == 0] = 1
    for i in range(nspec):
        API[i, :] = BSRk[i, :] * normAPI[i] + 1e-18
        API[i, :] = API[i, :] * signFI1[i] if API_changed_sign else API[i, :]
        BdS[i, :] = BdS[i, :] * signFI1[i]
        ALPHAs[:, i] = ALPHAs[:, i] * signFI1[i]
        BETAs[i, :] = BETAs[i, :] * signFI1[i]
        Fi[i] = Fi[i] * signFI1[i]
    
    return API, Fi, BdS, ALPHAs, BETAs


##############################################################################################
def calculate_TPI(nspec, kreac, yy, evalRI, ALPHA, BETA):
    # Assuming module_SRgR is implemented elsewhere in Python and returns SS, RR, gradRR
    SS = model.stoicVect(nspec, kreac)
    RR = eval_RR(yy, model.parameters(yy))
    gradRR = eval_gradR(yy, model.parameters(yy))
    BdS = BETA @ SS

    # Initialize BgSRkA with zeros
    BgSRkA = np.zeros((nspec, kreac))
    
    for j in range(kreac):
        gradSRk = np.outer(SS[:, j], gradRR[j, :])  # grad(S_k * R^k)
        for i in range(nspec):
            if evalRI[i, 1] == 0:  # FOR REAL EIGENVALUES
                BgSRkA[i, j] = BETA[i, :] @ gradSRk @ ALPHA[:, i] + 1e-18
            else:
                if i<nspec-1:
                    if evalRI[i, 1] == -evalRI[i + 1, 1]:  # FOR COMPLEX PAIR OF EIGENVALUES
                        b1Ja1 = BETA[i, :] @ gradSRk @ ALPHA[:, i]
                        b1Ja2 = BETA[i, :] @ gradSRk @ ALPHA[:, i + 1]
                        b2Ja1 = BETA[i + 1, :] @ gradSRk @ ALPHA[:, i]
                        b2Ja2 = BETA[i + 1, :] @ gradSRk @ ALPHA[:, i + 1]
                        BgSRkA[i, j] = 0.5 * (b1Ja1 + b2Ja2) + 1e-18
                        BgSRkA[i + 1, j] = 0.5 * (b1Ja2 - b2Ja1) + 1e-18

    # Calculate TPI
    TPI = np.zeros((nspec, kreac))
    for i in range(nspec):
        TPI[i, :] = BgSRkA[i, :] / np.sum(np.abs(BgSRkA[i, :]))

    return TPI, BdS

##############################################################################################
def calculate_Pointer(nspec, evalRI, ALPHA, BETA):
    Po = np.zeros((nspec, nspec))  # modes per species

    for i in range(nspec):
        if evalRI[i, 1] == 0:  # for real eigenvalues
            for j in range(nspec):
                Po[i, j] = BETA[i, j] * ALPHA[j, i]
        else:
            if i<nspec-1:
                if evalRI[i, 1] == -evalRI[i + 1, 1]:  # for complex pair of eigenvalues
                    for j in range(nspec):
                        Po[i, j]  = 0.5 * (BETA[i,j]*ALPHA[j,i] + BETA[i+1,j]*ALPHA[j,i+1])
                        Po[i+1,j] = 0.5 * (BETA[i+1,j]*ALPHA[j,i] - BETA[i,j]*ALPHA[j,i+1])
    return Po

##############################################################################################
def calculate_II(nspec, kreac, yy, ALPHA, BETA, NoEM =1, fast_species=[1],):
    # Initialize the Importance Index matrix
    II = np.zeros((nspec, kreac))
    

    fast_species_position = [x - 1 for x in fast_species]

    SS = model.stoicVect(nspec, kreac)
    RR = eval_RR(yy, model.parameters(yy))
    
    # Create an array of columns to include
    all_columns = np.arange(ALPHA.shape[1])
    selected_columns = np.setdiff1d(all_columns, fast_species_position)
    
    # Select the specified columns from ALPHA and BETA
    ALPHAslow = ALPHA[:, selected_columns]
    BETAslow = BETA[selected_columns, :]
    
    # Compute the product of slow modes with SS
    AsBsS = ALPHAslow @ BETAslow @ SS
    
    # Calculate the Importance Index
    for i in range(nspec):
        for j in range(kreac):
            II[i, j] = AsBsS[i, j] * RR[j] + 1e-18  # Adding a small number to avoid NaNs
        # Normalize the row values of II
        II[i, :] = II[i, :]/ np.sum(np.abs(II[i, :]))    
    return II



























################################################################################################
################################################################################################
################################################################################################
#-----------------------------------------------------------------------------------------------
def plot_rates(t, RR, kreac, figs, figs_format, time_unit, xlim, ylim=[1e-6, 1e3], log=[False, False], interactive=False):
    if not os.path.exists(figs): os.makedirs(figs)
    colors=colors_fun_reac(kreac)
    x_axis_type = 'log' if log[0] else 'linear'
    y_axis_type = 'log' if log[1] else 'linear'

    if interactive:
        fig = go.Figure()
        for i in range(1, kreac+1):
            fig.add_trace(go.Scatter(x=t, y=RR[i,:], line=dict(color=colors[i-1]), name=f'R{i}'))
        
        fig.update_layout(title='Reaction Rates', 
                            xaxis_title=f'Time ({time_unit})', 
                            yaxis_title='R', 
                            xaxis=dict(range=[np.log10(xlim[0]), np.log10(xlim[1])] if all(v is not None for v in xlim) else None),
                            yaxis=dict(range=[np.log10(ylim[0]), np.log10(ylim[1])] if all(v is not None for v in ylim) else None),
                            xaxis_type=x_axis_type,
                            yaxis_type=y_axis_type,
                            width=1000, height=650,
                         )
        fig.show()
        
    else:
        
        plt.figure(figsize=(6, 5))
        for i in range(1, kreac+1):  # Adjusted to loop through kreac
            plt.plot(t, RR[i,:], color=colors[i-1], marker='', linestyle='-', markersize=3, label=f'$R^{{{i}}}$')
        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18

        # plt.title('Reaction Rates')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel(r'$R^i$', rotation=0)
        plt.xscale(x_axis_type)
        plt.yscale(y_axis_type)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(loc="upper right", ncol=1)
        # plt.grid(True)
        plt.savefig(f'{figs}/Reaction_Rates.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

#-----------------------------------------------------------------------------------------------
def plot_timescales(t, timescales, exp_timescales, nspec, figs, figs_format, time_unit, xlim, ylim, nfast=1, log=[False, True], interactive=False):
    if not os.path.exists(figs): os.makedirs(figs)
    # colors = ['whitesmoke', 'gainsboro','lightgrey', 'darkgrey', 'grey']
    x_axis_type = 'log' if log[0] else 'linear'
    y_axis_type = 'log' if log[1] else 'linear'

    if interactive:
        # colors = [f'rgb({int(255*x)}, {int(255*x)}, {int(255*x)})' for x in np.linspace(0.8, 0.1, nspec)]
        colors = [f'rgb({int(255*x)}, {int(255*x)}, {int(255*x)})' for x in np.linspace(0.8, 0.1, nspec)][::-1]
        fig = go.Figure()
        # for i in range(nspec, nfast, -1):
        #     fig.add_trace(go.Scatter(x=t, y=timescales[i], line=dict(color='darkgrey'), name=rf'$\tau_{{{i}}}$'))
        
        slow_colors = colors[nfast:][::-1]
        for j, i in enumerate(range(nspec, nfast, -1)):
            fig.add_trace(go.Scatter(
                x=t, y=timescales[i], 
                line=dict(
                    color=slow_colors[j % len(colors)],
                    dash='dash',
                    width=2
                ),
                name=rf'$\tau_{{{i}}}$'
        ))

        fast_colors = colors[:nfast][::-1]
        for j, i in enumerate(range(nfast, 0, -1)):
        # for i in range(nfast, 0, -1):
            fig.add_trace(go.Scatter(x=t, y=timescales[i], 
                # line=dict(color='black'), 
                line=dict(
                    color=fast_colors[j % len(colors)],
                    dash='solid',        # <-- εδώ!
                    width=2
                ),
                name=rf'$\tau_{{{i}}}$'))
            
        if exp_timescales is not None:
            label_added = False
            for i in range(nspec, 0, -1):
                if np.any(exp_timescales[i] != 0):
                    non_zero = exp_timescales[i] != 0
                    indices  = np.where(non_zero)[0]
                    segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
            
                    for segment in segments:
                        fig.add_trace(go.Scatter(
                            x=   t[segment],
                            y=   exp_timescales[i][segment],
                            mode='lines',
                            line=dict(color='red'),
                            name=r'$\tau_e$',
                            legendgroup='tau_e',           # <-- all red traces share this group
                            showlegend=not label_added     # <-- only the first one shows in legend
                        ))
                        label_added = True



        fig.update_layout(title='Time scales', 
                            xaxis_title=f'Time ({time_unit})', 
                            yaxis_title=rf'$\tau_i ({{{time_unit}}})$',
                            xaxis_type =x_axis_type,
                            yaxis_type =y_axis_type,
                            # xaxis=dict(range=xlim),
                            # yaxis=dict(range=ylim),
                            xaxis=dict(range=[np.log10(xlim[0]), np.log10(xlim[1])] if all(v is not None for v in xlim) else None),
                            yaxis=dict(range=[np.log10(ylim[0]), np.log10(ylim[1])] if all(v is not None for v in ylim) else None),
                            width=1000, height=650,
                        )
        fig.show()
    else:
        colors = [str(x) for x in np.linspace(0.8, 0.1, nspec)][::-1]
        plt.figure(figsize=(6, 5))
        # for i in range(nspec, nfast, -1):  # Adjusted to loop through kreac
            # plt.plot(t, timescales[i], color='darkgrey', marker='', linestyle='--', markersize=3, label=f'$\\tau_{i}$')
        slow_colors = colors[nfast:][::-1]
        for j, i in enumerate(range(nspec, nfast, -1)):
            plt.plot(
                t, timescales[i], 
                color=slow_colors[j % len(colors)], 
                marker='', linestyle='--', markersize=3, 
                label=f'$\\tau_{i}$'
            )

        # for i in range(nfast,0, -1):  # Adjusted to loop through kreac
        fast_colors = colors[:nfast][::-1]
        for j, i in enumerate(range(nfast, 0, -1)):
            plt.plot(t, timescales[i], 
                color=fast_colors[j % len(colors)],
                marker='', linestyle='-', markersize=3, 
                label=f'$\\tau_{i}$'
            )

        if exp_timescales is not None:
            label_added = False  # Track if the label has been added
            for i in range(nspec, 0, -1):
                if np.any(exp_timescales[i] != 0):  # Check for non-zero values
                    non_zero = exp_timescales[i] != 0  # Boolean mask
                    indices = np.where(non_zero)[0]  # Indices of non-zero values
                    segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

                    for segment in segments:
                        plt.plot(t[segment], exp_timescales[i][segment], color='red', marker='', linestyle='-', 
                                 markersize=3, label='$\\tau_e$' if not label_added else "")
                        label_added = True  # Label added after the first plot

            

        nfast
        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18
    
        # plt.title('Time scales')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel(r'$\tau_{i}$ ' f'({time_unit})', rotation=0, labelpad=10)
        plt.yscale(y_axis_type)
        plt.xscale(x_axis_type)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(loc="upper right", ncol=1)
        # plt.grid(True)
        plt.savefig(f'{figs}/timescales.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

#------------------------------------------------------------------------------------
def plot_amplitudes(t, amplitudes, nspec, figs, figs_format, time_unit, xlim, ylim, nfast=1, positive=True, log=[False, True], interactive=False):
    if not os.path.exists(figs): os.makedirs(figs)
    # colors = ['grey', 'darkgrey', 'lightgrey', 'gainsboro', 'whitesmoke']
    # colors = ['whitesmoke', 'gainsboro','lightgrey', 'darkgrey', 'grey']
    x_axis_type = 'log' if log[0] else 'linear'
    y_axis_type = 'log' if log[1] else 'linear'

    if interactive:
        colors = [f'rgb({int(255*x)}, {int(255*x)}, {int(255*x)})' for x in np.linspace(0.8, 0.1, nspec)][::-1]
        fig = go.Figure()
        # for i in range(nspec, nfast, -1):
        slow_colors = colors[nfast:][::-1]
        for j, i in enumerate(range(nspec, nfast, -1)):
            fig.add_trace(go.Scatter(
                x=t, y=amplitudes[i,:], 
                # line=dict(color=colors[j % len(colors)]),
                line=dict(
                    color=slow_colors[j % len(colors)],
                    dash='dash',
                    width=2
                ),
                # line=dict(color=colors[j]), 
                name=f'f{i}'
        ))
            # fig.add_trace(go.Scatter(x=t, y=amplitudes[i,:], line=dict(color='darkgrey'), name=f'f{i}'))

        fast_colors = colors[:nfast][::-1]
        for j, i in enumerate(range(nfast, 0, -1)):
            fig.add_trace(go.Scatter(x=t, y=amplitudes[i,:], 
                line=dict(
                    color=fast_colors[j % len(colors)],
                    dash='solid',
                    width=2
                    ), 
                name=f'f{i}'))
        
        # axis_type = 'log' if log else 'linear'

        fig.update_layout(title='CSP Amplitudes', 
                            xaxis_title=f'Time ({time_unit})', 
                            yaxis_title='f', 
                            xaxis_type=x_axis_type,
                            yaxis_type=y_axis_type,
                            # xaxis=dict(range=xlim),
                            xaxis=dict(range=[np.log10(xlim[0]), np.log10(xlim[1])] if all(v is not None for v in xlim) else None),
                            yaxis=dict(range=[np.log10(ylim[0]), np.log10(ylim[1])] if all(v is not None for v in ylim) else None),
                             width=1000, height=650,
                        )
        fig.show()
    
    else:
        colors = [str(x) for x in np.linspace(0.8, 0.1, nspec)][::-1]
        plt.figure(figsize=(6, 5))
        # for i in range(nspec, nfast, -1):
        slow_colors = colors[nfast:][::-1]
        for j, i in enumerate(range(nspec, nfast, -1)):
            plt.plot(
                t, amplitudes[i,:], 
                color=slow_colors[j % len(colors)], 
                marker='', linestyle='--', markersize=3, 
                label=f'$f^{i}$'
            )
            # plt.plot(t, amplitudes[i,:], color='darkgrey', marker='', linestyle='--', markersize=3, label=f'$f^{i}$')
        fast_colors = colors[:nfast][::-1]
        for j, i in enumerate(range(nfast, 0, -1)):
            plt.plot(t, amplitudes[i,:], 
                color=fast_colors[j % len(colors)], 
                marker='', linestyle='-', markersize=3, 
                label=f'$f^{i}$'
                )

        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18

        # plt.title('CSP Amplitudes')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel('$f^i$', rotation = 0, labelpad=20)
        plt.yscale(y_axis_type)
        plt.xscale(x_axis_type)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(loc="upper right", ncol=1)
        # plt.grid(True)
        plt.savefig(f'{figs}/amplitudes.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

#-----------------------------------------------------------------------------------------------
def plot_Po(t, Pointer, nspec, vars, figs, figs_format, time_unit, xlim, ylim, mode=1, interactive=True):
    if not os.path.exists(figs): os.makedirs(figs)
    colors=colors_fun(nspec)
    if interactive:
        fig = go.Figure()
        for i in range(1, nspec+1):
            fig.add_trace(go.Scatter(x=t, y=Pointer[mode,i, :], line=dict(color=colors[i-1]), name=f'{vars[i-1]}'))
    
        fig.update_layout(title=f'CSP Pointer of Mode {mode}', 
                            xaxis_title=f'Time ({time_unit})', 
                            yaxis_title='Po', 
                            xaxis=dict(range=xlim),
                            yaxis=dict(range=ylim),
                            width=1000, height=650,
                        )
        fig.show()

    else:
        plt.figure(figsize=(6, 5))
        for i in range(1, nspec+1):  # Adjusted to loop through kreac
            plt.plot(t, Pointer[mode,i, :], color=colors[i-1], marker='', linestyle='-', markersize=3, label=f'${vars[i-1]}$')

        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18

        # plt.title(f'CSP Pointer of Mode {mode}')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel(f'$Po^{mode}$', rotation=0)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(loc="upper right", ncol=1)
        # plt.grid(True)
        plt.savefig(f'{figs}/Po_{mode}.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

#-----------------------------------------------------------------------------------------------
def plot_TPIs(t, TPIs, kreac, figs, figs_format, time_unit, xlim, ylim, mode, interactive=True):
    if not os.path.exists(figs): os.makedirs(figs)
    colors=colors_fun_reac(kreac)
    if interactive:
        fig = go.Figure()
        for i in range(1,kreac+1):
            fig.add_trace(go.Scatter(x=t, y=TPIs[mode, i, :], line=dict(color=colors[i-1]), name=f'R{i}'))
        
        fig.update_layout(title=f'TPIs of Mode {mode}', 
                            xaxis_title=f'Time ({time_unit})', 
                            yaxis_title='TPI',                          
                            xaxis=dict(range=xlim),
                            yaxis=dict(range=ylim),
                            width=1000, height=650,
                        )
        fig.show()
    
    else:
        plt.figure(figsize=(6, 5))
        for i in range(1, kreac+1):  # Adjusted to loop through kreac
            plt.plot(t, TPIs[mode, i, :], color=colors[i-1], marker='', linestyle='-', markersize=3, label=f'$R^{{{i}}}$')
    
        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18

        # plt.title(f'TPIs of Mode {mode}')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel(f'TPI$^{mode}$')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(loc="upper right", ncol=1)
        # plt.grid(True)
        plt.savefig(f'{figs}/TPI_{mode}.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

#------------------------------------------------------------------------------------
def plot_APIs(t, APIs, kreac, figs, figs_format, time_unit, xlim, ylim, mode, positive=True, interactive=True):
    if not os.path.exists(figs): os.makedirs(figs)
    colors=colors_fun_reac(kreac)
    if interactive:
        fig = go.Figure()
        for i in range(1,kreac+1):
            fig.add_trace(go.Scatter(x=t, y=np.abs(APIs[mode, i, :]) if positive else APIs[mode, i, :], line=dict(color=colors[i-1]), name=f'R{i}'))
        
        fig.update_layout(title=f'APIs of Mode {mode}', xaxis_title=f'Time ({time_unit})', yaxis_title='|API|' if positive else 'API',                          width=1000, height=650,
            )
        fig.show()
    else:
        plt.figure(figsize=(6, 5))
        for i in range(1, kreac+1):  # Adjusted to loop through kreac
            plt.plot(t, np.abs(APIs[mode, i, :]) if positive else APIs[mode, i, :], color=colors[i-1], marker='', linestyle='-', markersize=3, label=f'$R^{{{i}}}$')
    
        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18

        # plt.title(f'APIs of Mode {mode}')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel(f'|API$^{mode}$|' if positive else f'API$^{mode}$')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(loc="upper right", ncol=1)
        # plt.grid(True)
        plt.savefig(f'{figs}/API_{mode}.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

#-----------------------------------------------------------------------------------------------
def plot_IIs(t, IIs, kreac, figs, figs_format, vars, time_unit, xlim, ylim, variable, interactive=True):
    if not os.path.exists(figs): os.makedirs(figs)
    colors=colors_fun_reac(kreac)
    
    if interactive:
        fig = go.Figure()
        for i in range(1, kreac+1):
            fig.add_trace(go.Scatter(x=t, y=IIs[variable, i, :], line=dict(color=colors[i-1]), name=f'R{i}'))

        fig.update_layout(title=f'IIs of variable {variable}', 
                            xaxis_title=f'Time ({time_unit})', 
                            yaxis_title='II',
                            xaxis=dict(range=xlim),
                            yaxis=dict(range=ylim),
                            width=1000, height=650,
                         )
        fig.show()
        
    else:
        
        plt.figure(figsize=(6, 5))
        for i in range(1, kreac+1):  # Adjusted to loop through kreac
            plt.plot(t, IIs[variable, i, :], color=colors[i-1], marker='', linestyle='-', markersize=3, label=f'$R^{{{i}}}$')

        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18
    
        # plt.title(f'IIs of variable {variable}')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel(f'II$^{vars[variable-1]}$')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(loc="upper right", ncol=1)
        # plt.grid(True)
        plt.savefig(f'{figs}/II_{variable}.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

#-----------------------------------------------------------------------------------------------
def write_rates(t,Rates,kreac,results_path='results', text_format='.txt'):
    R_columns = ['R{}'.format(i) for i in range(1, kreac+1)]
    df_R = pd.DataFrame(Rates[1:,:].T, columns=R_columns)
    df_R.insert(0, 't', t)  # Insert the time vector as the first column

    if not os.path.exists(results_path): os.makedirs(results_path)
    df_R.to_csv(f"{results_path}/Rates.{text_format}", sep="\t", float_format='%.8e', index=False)
    del df_R
    import gc
    gc.collect()
    
#-----------------------------------------------------------------------------------------------
def write_eigens(t, eigens, nspec, results_path='results', text_format='.txt'):
    t_columns = [f'Re_{i//2+1}' if i % 2 == 0 else f'Im_{i//2+1}' for i in range(2 * nspec)]
    df_t = pd.DataFrame(eigens[1:,:].T, columns=t_columns)
    df_t.insert(0, 't', t)  # Insert the time vector as the first column

    if not os.path.exists(f'{results_path}/diagn'): os.makedirs(f'{results_path}/diagn')
    df_t.to_csv(f"{results_path}/diagn/eigens.{text_format}", sep="\t", float_format='%.8e', index=False)
    del df_t
    import gc
    gc.collect()
#-----------------------------------------------------------------------------------------------
def write_timescales(t,timescales,nspec, results_path='results', text_format='.txt'):
    t_columns = ['tau_{}'.format(i) for i in range(1, nspec+1)]
    df_t = pd.DataFrame(timescales[1:,:].T, columns=t_columns)
    df_t.insert(0, 't', t)  # Insert the time vector as the first column

    if not os.path.exists(f'{results_path}/diagn'): os.makedirs(f'{results_path}/diagn')
    df_t.to_csv(f"{results_path}/diagn/timescales.{text_format}", sep="\t", float_format='%.8e', index=False)
    del df_t
    import gc
    gc.collect()

#-----------------------------------------------------------------------------------------------
def write_amplitudes(t,amplitudes,nspec,results_path='results',text_format='.txt'):
    f_columns = ['f{}'.format(i) for i in range(1, nspec+1)]
    df_f = pd.DataFrame(amplitudes[1:,:].T, columns=f_columns)
    df_f.insert(0, 't', t)  # Insert the time vector as the first column
    
    if not os.path.exists(f'{results_path}/diagn'): os.makedirs(f'{results_path}/diagn')
    df_f.to_csv(f"{results_path}/diagn/amplitudes.{text_format}", sep="\t", float_format='%.8e', index=False)
    del df_f
    import gc
    gc.collect()

#-----------------------------------------------------------------------------------------------
def write_pointers(t,Pointer,nspec,results_path='results',text_format='.txt'):
    for m in range(1, nspec+1):
        y_columns = ['y{}'.format(i) for i in range(1, nspec+1)]

        df_Po = pd.DataFrame(Pointer[1:, m, :].T, columns=y_columns)
        df_Po.insert(0, 't', t)  # Insert the time vector as the first column
    
        if not os.path.exists(f'{results_path}/diagn/pointers'):
            os.makedirs(f'{results_path}/diagn/pointers')
        filename = f"{results_path}/diagn/pointers/Po{m}.{text_format}"
        df_Po.to_csv(filename, sep="\t", float_format='%.8e', index=False)
    del df_Po
    import gc
    gc.collect()

#-----------------------------------------------------------------------------------------------
def write_TPIs(t,TPIs,nspec,kreac,results_path='results',text_format='.txt'):
    for m in range(1, nspec+1):
        R_columns = ['R{}'.format(i) for i in range(1, kreac+1)]
    
        df_TPI = pd.DataFrame(TPIs.T[:,1:,m], columns=R_columns)
        df_TPI.insert(0, 't', t)  # Insert the time vector as the first column
        
        if not os.path.exists(f'{results_path}/diagn/TPIs'):
            os.makedirs(f'{results_path}/diagn/TPIs')
        filename = f"{results_path}/diagn/TPIs/TPI{m}.{text_format}"
        df_TPI.to_csv(filename, sep="\t", float_format='%.8e', index=False)
    del df_TPI
    import gc
    gc.collect()

#-----------------------------------------------------------------------------------------------
def write_APIs(t,APIs,nspec,kreac,results_path='results',text_format='.txt'):
    for m in range(1, nspec+1):
        R_columns = ['R{}'.format(i) for i in range(1, kreac+1)]
    
        df_API = pd.DataFrame(APIs.T[:,1:,m], columns=R_columns)
        df_API.insert(0, 't', t)  # Insert the time vector as the first column
        
        if not os.path.exists(f'{results_path}/diagn/APIs'):
            os.makedirs(f'{results_path}/diagn/APIs')
        filename = f"{results_path}/diagn/APIs/API{m}.{text_format}"
        df_API.to_csv(filename, sep="\t", float_format='%.8e', index=False)
    del df_API
    import gc
    gc.collect()

#-----------------------------------------------------------------------------------------------
def write_IIs(t,IIs,nspec,kreac,results_path ='results',text_format='.txt'):
    for m in range(1, nspec+1):
        R_columns = ['R{}'.format(i) for i in range(1, kreac+1)]
    
        df_II = pd.DataFrame(IIs.T[:,1:,m], columns=R_columns)
        df_II.insert(0, 't', t)  # Insert the time vector as the first column
        
        if not os.path.exists(f'{results_path}/diagn/IIs'):
            os.makedirs(f'{results_path}/diagn/IIs')
        filename = f"{results_path}/diagn/IIs/II{m}.{text_format}"
        df_II.to_csv(filename, sep="\t", float_format='%.8e', index=False)
    del df_II
    import gc
    gc.collect()
