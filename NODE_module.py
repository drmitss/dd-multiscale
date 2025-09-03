# NODE module
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.optim.lr_scheduler as lr_scheduler

# ANSI color codes for progress bar
BLUE = "\033[94m"
RESET = "\033[0m"
tqdm._instances.clear()

data = None

class ODEFunc(nn.Module):
    """
    Neural ODE right-hand side: dy/dt = f(y; t)

    Args:
        nspec (int): dimensionality of the system.
        hidden_dims (list of int): sizes of hidden layers.
    """
    def __init__(self, nspec, hidden_dims=[100, 100, 50]):
        super(ODEFunc, self).__init__()
        layers = []
        in_dim = nspec
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, nspec))
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        return self.net(y)

################################################################################
def train_node(
    ode_func,
    t,
    y0,
    y_true,
    epochs=200,
    early_stop_patience=100,
    lr=1e-3,
    weight_decay=1e-4,
    verbose=True,
    rtol=1e-6,
    atol=1e-6,
    min_delta=1e-7,
    max_grad_norm=1.0,
    use_scheduler=True
):
    """
    Train a Neural ODE to match the given trajectory.

    Args:
        ode_func (ODEFunc): an instance of the ODEFunc class.
        t (array-like, shape (T,)): time points.
        y0 (array-like, shape (n,)): initial state at t[0].
        y_true (array-like, shape (T, n)): ground truth trajectory.
        epochs (int): number of training epochs.
        lr (float): optimizer learning rate.
        verbose (bool): if True, show a tqdm bar and loss updates.
        rtol (float): relative tolerance for adaptive solver on CPU/CUDA.
        atol (float): absolute tolerance for adaptive solver on CPU/CUDA.
        early_stop_patience (int): stop if no improvement for this many epochs.
        min_delta (float): minimal relative loss improvement to reset patience.

    Returns:
        np.ndarray: predicted trajectory of shape (T, n).
    """
    # Choose device & solver
    if torch.backends.mps.is_available():
        device, solver, solver_kwargs = torch.device('mps'), 'rk4', {}
    elif torch.cuda.is_available():
        device, solver, solver_kwargs = torch.device('cuda'), 'dopri5', {'rtol': rtol, 'atol': atol}
    else:
        device, solver, solver_kwargs = torch.device('cpu'), 'dopri5', {'rtol': rtol, 'atol': atol}

    ode_func.to(device)

    # prepare tensors
    t_train = torch.tensor(t, dtype=torch.float32, device=device)
    y0_t    = torch.tensor(y0, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_true, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(ode_func.parameters(), lr=lr, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    loss_fn = lambda pred, true: torch.mean((pred - true) ** 2)

    # early‐stop bookkeeping
    best_loss = float('inf')
    epochs_no_improve = 0

    ode_func.train()
    # wrap epochs in tqdm if verbose
    tqdm._instances.clear()
    epoch_iter = range(1, epochs + 1)
    if verbose:
        epoch_iter = tqdm(
            epoch_iter,
            desc='Training Neural ODE',
            dynamic_ncols=True,
            leave=False,
            bar_format=f"{{l_bar}}{BLUE}{{bar}}{RESET} {{n_fmt}}/{{total_fmt}} {{elapsed}}",
            ascii='░█'
        )
    for epoch in epoch_iter:
        optimizer.zero_grad()
        y_pred = odeint(ode_func, y0_t, t_train, method=solver, **solver_kwargs)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        clip_grad_norm_(ode_func.parameters(), max_grad_norm)
        optimizer.step()
        if use_scheduler:
            scheduler.step()

        current_loss = loss.item()
        if verbose:
            epoch_iter.set_postfix({'loss': f'{current_loss:.2e}'})

        if early_stop_patience is not None:
            # Only check early stopping if patience is set
            if best_loss - current_loss > min_delta * best_loss:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                if verbose:
                    print(f"\n⏹️ Early stopping at epoch {epoch} (no improvement in {early_stop_patience} epochs)")
                break

    # final evaluation
    ode_func.eval()
    with torch.no_grad():
        y_pred = odeint(ode_func, y0_t, t_train, method=solver, **solver_kwargs)

    return y_pred.cpu().numpy()


################################################################################
def extrapolate_trajectory(
    ode_func,
    y0,
    t0=0.0,
    tf=20.0,
    points=20000,
    device=None,
    rtol=1e-6,
    atol=1e-7
):
    """
    Integrate the trained ODE function to extrapolate the trajectory.

    Args:
        ode_func (ODEFunc): trained ODEFunc neural network.
        y0 (np.ndarray or torch.Tensor): initial state, shape (n,) or (1, n).
        t0 (float): start time.
        tf (float): end time.
        points (int): number of time points for integration.
        device (str or torch.device, optional): device to use (auto-detect if None).
        rtol (float, optional): relative tolerance for the ODE solver.
            Smaller values mean stricter error control, but may increase computation time.
            **Note:** On Apple Silicon/MPS, use `float32` precision (e.g., `rtol=np.float32(1e-6)`).
        atol (float, optional): absolute tolerance for the ODE solver.
            Like `rtol`, smaller values result in stricter error control.
            **Note:** On Apple Silicon/MPS, use `float32` precision (e.g., `atol=np.float32(1e-7)`).

    Returns:
        t_extrapolate (np.ndarray): time vector, shape (points,)
        y_extrapolate (np.ndarray): predicted trajectory, shape (points, n)
    """
    if device is None:
        # Auto-select device as in your train_node function
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # Prepare time vector and tensors
    t_extrapolate = np.linspace(t0, tf, points)
    t_extrapolate_tensor = torch.tensor(t_extrapolate, dtype=torch.float32, device=device)

    # Initial state as tensor (ensure correct device & shape)
    if isinstance(y0, np.ndarray):
        y0_torch = torch.tensor(y0, dtype=torch.float32, device=device)
    else:
        y0_torch = y0.detach().to(device).to(torch.float32)

    if y0_torch.ndim == 1:
        pass  # shape (n,)
    elif y0_torch.ndim == 2 and y0_torch.shape[0] == 1:
        y0_torch = y0_torch[0]
    else:
        raise ValueError("y0 must be shape (n,) or (1, n)")

    ode_func = ode_func.to(device)
    ode_func.eval()
    # with torch.no_grad():
    #     y_extrapolate = odeint(
    #         ode_func,
    #         y0_torch,
    #         t_extrapolate_tensor,
    #         method='rk4'
    #     )
    with tqdm(total=1, desc='Extrapolating trajectory', bar_format='{l_bar}{bar} {elapsed}') as pbar:
        with torch.no_grad():
            y_extrapolate = odeint(
                ode_func,
                y0_torch,
                t_extrapolate_tensor,
                method='rk4'
            )
        pbar.update(1)
    return t_extrapolate, y_extrapolate.cpu().numpy()

################################################################################    
def compute_derivatives(
    ode_func,
    y_pred
):
    """
    Compute dy/dt = f(y) along a predicted trajectory.

    Args:
        ode_func (ODEFunc): trained ODEFunc instance.
        y_pred (np.ndarray or torch.Tensor of shape (T, n)): trajectory.

    Returns:
        np.ndarray: derivatives at each time, shape (T, n).
    """
    device = next(ode_func.parameters()).device

    if not isinstance(y_pred, torch.Tensor):
        y_t = torch.tensor(y_pred, dtype=torch.float32, device=device)
    else:
        y_t = y_pred.detach().to(device)

    ode_func.eval()
    with torch.no_grad():
        try:
            deriv = ode_func(0.0, y_t)
        except RuntimeError:
            out = []
            for i in range(y_t.shape[0]):
                out.append(ode_func(0.0, y_t[i]))
            deriv = torch.stack(out)

    return deriv.cpu().numpy()