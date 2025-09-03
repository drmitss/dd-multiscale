# model_module.py
import os
import numpy as np
import sympy as smp
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Define system size ---------------------------------------------------------------------
global nSpec, kReac
nSpec = 2      # Number of species/variables
kReac = 3      # Number of reactions

# Define system parameters ---------------------------------------------------------------
def parameters(y):
    global k_1f, k_1b, k_2, e_0 
    global params
    k_1f = 10000   # formation rate constant of the enzyme-substrate complex
    k_1b = 0.01    # dissociation rate constant
    k_2 = 0.1      # catalysis rate constant
    e_0 = 10       # total enzyme concentration

    params = k_1f, k_1b, k_2, e_0
    return params



# Define system parameters symbolically --------------------------------------------------
def var_names():
    var_names = ['c', 's']
    return var_names

def param_names():
    param_names = ['k_1f', 'k_1b', 'k_2', 'e_0']
    return param_names

def create_symbolic_parameters():
    names = param_names()
    symbols = smp.symbols(names)
    sym_dict = dict(zip(names, symbols))
    globals().update(sym_dict)
    return names, symbols
    

# Define the ODE system analytically -----------------------------------------------------
def system_of_eqs(t, y):
    k_1f, k_1b, k_2, e_0 = parameters(y)
    c, s = y
    
    dydt = [ k_1f*(e_0 - c)*s -k_1b*c -k_2*c,
            -k_1f*(e_0 - c)*s +k_1b*c]
    return dydt

def slow_system_of_eqs_sQSSA(t, y):
    c, s = y
    k_1f, k_1b, k_2, e_0 = parameters(y)

    K=k_2/k_1f
    K_R = k_1b/k_1f
    K_M = K_R + K

    dydt = [-(k_2*(e_0-c)**2*c)/(K_M*e_0), 
            -1.0*k_2*c]
    return dydt

def slow_system_of_eqs_rQSSA(t, y):
    c, s = y
    k_1f, k_1b, k_2, e_0 = parameters(y)

    K=k_2/k_1f
    K_R = k_1b/k_1f

   
    dydt = [-1.0*k_2*c, 
        -(k_2*(K_R+s)*s)/K_R]
    return dydt

def slow_system_of_eqs_PEA(t, y):
    c, s = y
    k_1f, k_1b, k_2, e_0 = parameters(y)

    K_R = k_1b/k_1f
    nu  = (e_0-c)/(K_R+s)

   
    dydt = [-nu/(1+nu)*k_2*c, 
            -1/(1+nu)*k_2*c]
    return dydt

# Based on g = S.R    
# Define the ODE system based on g = S.R -------------------------------------------------
def RHS(t, y):
    params = parameters(y)         # Get numeric parameters
    RR_vec = RR(y, params)         # Use the lambdified RR from symbolic model
    SS = stoicVect(nSpec, kReac)
    return SS.dot(np.array(RR_vec))

    
# Define stoichiometry matrix ------------------------------------------------------------
def stoicVect(nSpec, kReac):
    global SS
    SS = np.array([
    [+1, -1, -1],
    [-1, +1,  0]
    ])
    return SS

# Define reaction rates ------------------------------------------------------------------
def calc_R(y, kreac):
    create_symbolic_parameters()
    RR_list = [None] * (kreac+1)

    [y1,y2] = y;

    RR_list[1] = k_1f*(e_0 - y1)*y2;
    RR_list[2] = k_1b*y1; 
    RR_list[3] = k_2*y1; 
    return RR_list

# Define reaction rates gradient ---------------------------------------------------------
def calc_gradR(RR_list, y):
    create_symbolic_parameters()

    gradR_list = []
    for i in range(1, len(RR_list)):
        gradR_row = [smp.diff(RR_list[i], yi) for yi in y]
        gradR_list.append(gradR_row)
    return gradR_list



# Add noise to the data ------------------------------------------------------------------
from sklearn.metrics import root_mean_squared_error
def add_noise(y, noise_type, percentage):
    """
    Adds noise to the input data.

    Parameters:
    y (numpy array): Input data.
    noise_type (str): Type of noise to add. Can be 'additive' or 'multiplicative'.
    percentage (float): Percentage of noise to add.

    Returns:
    y_noise (numpy array): Noisy data.

    Example usage:
    np.random.seed(42)
    y = np.random.rand(100, 3)  # example data
    y_noise_additive = add_noise(y, 'additive', 2)
    y_noise_multiplicative = add_noise(y, 'multiplicative', 1)

    """
    rmse = root_mean_squared_error(y, np.zeros(y.shape))
    noise_scale = rmse * (percentage / 100)

    if noise_type == 'additive':
        y_noise = y + np.random.normal(0, noise_scale, y.shape)
    elif noise_type == 'multiplicative':
        y_noise = y * (1 + np.random.normal(0, noise_scale, y.shape))
    else:
        raise ValueError("Invalid noise type. Choose 'additive' or 'multiplicative'.")

    return y_noise




# Initiate the symbolic model ------------------------------------------------------------
def init_symbolic_model(nspec,kreac):
    global R_symbolic, gradR_symbolic, SS, g_symbolic, J_symbolic_R, J_symbolic
    global RR, gradR, SS, g, jac, jac_R, RR2, symbolic_parameters

    J_symbolic_R = []
    J_symbolic =[]
    y_sol = [smp.symbols(f'y{i+1}') for i in range(nspec)]
    create_symbolic_parameters()

    R_symbolic = calc_R(y_sol, kreac)
    gradR_symbolic = calc_gradR(R_symbolic, y_sol)
    SS = stoicVect(nspec, kreac)  # Example values, adjust as necessary

    g_symbolic = list(SS.dot(R_symbolic[1:]))
    for k in np.arange(kreac):
        J_symbolic_R.append(list(np.tensordot(SS.T[k],gradR_symbolic[k], axes=0)))

    for n in np.arange(nspec):
        J_symbolic.append(list(SS.dot(gradR_symbolic)[n]))

    for n in np.arange(nspec):
        for k in np.arange(kreac):
            J_symbolic_R[k][n] = list(J_symbolic_R[k][n])

    param_names_list, symbolic_parameters = create_symbolic_parameters()

    RR  = smp.lambdify([y_sol, symbolic_parameters], R_symbolic[1:], modules='numpy')
    gradR = smp.lambdify([y_sol, symbolic_parameters], gradR_symbolic, modules='numpy')
    g   = smp.lambdify([y_sol, symbolic_parameters], g_symbolic, 'numpy')
    jac = smp.lambdify([y_sol, symbolic_parameters], J_symbolic, 'numpy')
    jac_R = smp.lambdify([y_sol, symbolic_parameters], J_symbolic_R, 'numpy')

    return RR,  gradR , SS, g, g_symbolic, jac, jac_R , J_symbolic

def eval_RR(y, parameters):
    return np.array(RR(y, parameters))
def eval_gradR(y, parameters):
    return np.array(gradR(y, parameters))
def eval_g(y, parameters):
    return np.array(g(y, parameters))
def eval_jac(y, parameters):
    return np.array(jac(y, parameters))
def eval_jac_R(y, parameters):
    return np.array(jac_R(y, parameters))

    
##########################################################################################
def colors_fun(size):
    colors = ['red', 'blue', 'green', 'orange', 'black', 'purple', 'brown', 'pink', 'grey', 'navy', 'maroon', 'crimson', 'teal', 'violet', 'salmon', 'turquoise', 'wheat', 'tomato', 'lime', 'olive', 'chocolate', 'gold', 'indigo', 'beige', 'lavender', 'magenta', 'cyan']
    if size > len(colors): colors += colors * (size // len(colors))
    return colors

def colors_fun_reac(size):
    colors = ['salmon', 'turquoise', 'violet', 'indianred', 'darkgreen', 'crimson', 'lime', 'olive', 'cyan', 'chocolate', 'gold', 'indigo',  'magenta', 'teal', 'beige','black', 'purple', 'brown', 'pink', 'navy', 'maroon' ]
    if size > len(colors): colors += colors * (size // len(colors))
    return colors

# ----------------------------------------------------------------------------------------------------------------------
def plot_solution(t, y, nspec, figs, figs_format, vars, time_unit, xlim, ylim, log=True, interactive=False):
    if not os.path.exists(figs): os.makedirs(figs)
    colors = colors_fun(nspec)
    if nspec > len(colors):
        colors += colors * (nspec // len(colors))
    axis_type = 'log' if log else 'linear'
    
    if interactive:
        fig = go.Figure()
        # Add a line for each species
        for i in range(1, nspec+1):
            fig.add_trace(go.Scatter(x=t, y=y[i], mode='lines', line=dict(color=colors[i-1]), name=vars[i-1]))

        # Update the layout
        fig.update_layout(
            title='Model solution',
            xaxis_title=f'Time ({time_unit})',
            yaxis_title='y',
            xaxis=dict(range=xlim),
            yaxis=dict(range=ylim),
            yaxis_type= axis_type,
            width=1000,  height=650,
            legend_title="Variables"
        )
        # Show the figure
        fig.show()

    else:

        plt.figure(figsize=(7, 5))
        for i in range(1,nspec+1):
            plt.plot(t, y[i], color=colors[i-1], marker='', linestyle='-', markersize=3, label=f'${vars[i-1]}$(t)')
        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18
        # plt.title('Model solution')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel('$y_i$', rotation=0)
        plt.yscale(axis_type)
        # plt.legend()
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        plt.legend(loc="upper right", ncol=1)
        # plt.grid(True)
        plt.savefig(f'{figs}/solution.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

# ----------------------------------------------------------------------------------------------------------------------
def plot_rates(t, RR, kreac, figs, figs_format, time_unit, xlim, ylim=[1e-6, 1e3], log=True, interactive=False):
    if not os.path.exists(figs): os.makedirs(figs)
    colors=colors_fun_reac(kreac)
    axis_type = 'log' if log else 'linear'

    if interactive:
        fig = go.Figure()
        for i in range(1, kreac+1):
            fig.add_trace(go.Scatter(x=t, y=RR[i,:], line=dict(color=colors[i-1]), name=f'R{i}'))
        
        fig.update_layout(title='Reaction Rates', 
                            xaxis_title=f'Time ({time_unit})', 
                            yaxis_title='R', 
                            xaxis=dict(range=xlim),
                            yaxis=dict(range=[np.log10(ylim[0]), np.log10(ylim[1])] if all(v is not None for v in ylim) else None),
                            yaxis_type=axis_type,
                            width=1000, height=650,
                         )
        fig.show()
        
    else:
        
        plt.figure(figsize=(7, 5))
        for i in range(1, kreac+1):  # Adjusted to loop through kreac
            plt.plot(t, RR[i,:], color=colors[i-1], marker='', linestyle='-', markersize=3, label=f'$R^{{{i}}}$')
        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18

        # plt.title('Reaction Rates')
        plt.xlabel(f'Time ({time_unit})')
        plt.ylabel(r'$R^i$', rotation=0)
        plt.yscale(axis_type)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(loc="upper right", ncol=1)
        # plt.grid(True)
        plt.savefig(f'{figs}/Reaction_Rates.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

# ----------------------------------------------------------------------------------------------------------------------
def plot_phase(y, figs, figs_format, vars, pos, nskip, interactive=False):
    if not os.path.exists(figs): os.makedirs(figs)
    if interactive:
        # Create a single interactive 2D scatter plot for y1 vs y2
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y[pos[0]][::nskip], y=y[pos[1]][::nskip],
            mode='lines+markers',
            marker=dict(size=10), line=dict(width=2)
        ))
        
        fig.update_layout(
            width=1000, height=650,
            xaxis_title=vars[pos[0]-1],
            yaxis_title=vars[pos[1]-1]
        )
        
        fig.show()
    else:
        # Create a single non-interactive 2D plot for y1 vs y2
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        ax.plot(y[pos[0]][::nskip], y[pos[1]][::nskip], 'r-', marker='o', linewidth=2)
        
        ax.set_xlabel(f'${vars[pos[0]-1]}$')
        ax.set_ylabel(f'${vars[pos[1]-1]}$', rotation =0)
        
        plt.savefig(f'{figs}/phase_space.{figs_format}',dpi=400, bbox_inches='tight')
        plt.show()

# ----------------------------------------------------------------------------------------------------------------------
def write_solution(t,y,nspec, results_path='results', text_format='.txt'):
    column_names = ['y{}'.format(i) for i in range(1, nspec+1)]
    df_sol = pd.DataFrame(y[1:].T, columns=column_names)
    df_sol.insert(0, 't', t)  # Insert the time vector as the first column
    
    if not os.path.exists(results_path): os.makedirs(results_path)
    df_sol.to_csv(f"{results_path}/sol.{text_format}", sep="\t", float_format='%.8e', index=False)
    del df_sol
    import gc
    gc.collect()

# ----------------------------------------------------------------------------------------------------------------------
def read_solution(file_path, nspec):
    """
    Reads a solution file and returns time vector and species matrix.

    Args:
        file_path (str): Path to the solution file.
        nspec (int): Number of species (columns excluding time).

    Returns:
        t (np.ndarray): Time vector of shape (N,)
        x (np.ndarray): Species matrix of shape (N, nspec)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, sep="\t")
    
    expected_columns = ['t'] + [f'y{i}' for i in range(1, nspec+1)]
    if list(df.columns) != expected_columns:
        raise ValueError(f"File columns do not match expected format: {expected_columns}")

    t = df['t'].values
    x = df.iloc[:, 1:].values  # All species columns

    return t, x


# ----------------------------------------------------------------------------------------------------------------------
def write_rates(t,Rates,kreac,results_path = 'results', text_format='.txt'):
    R_columns = ['R{}'.format(i) for i in range(1, kreac+1)]
    df_R = pd.DataFrame(Rates[1:,:].T, columns=R_columns)
    df_R.insert(0, 't', t)  # Insert the time vector as the first column

    if not os.path.exists(results_path): os.makedirs(results_path)
    df_R.to_csv(f"{results_path}/Rates.{text_format}", sep="\t", float_format='%.8e', index=False)
    del df_R
    import gc
    gc.collect()
