import warnings
import numpy as np
from pysindy import SINDy, WeakPDELibrary, SR3, STLSQ
from sklearn.metrics import r2_score

def run_weak_sindy(t, x, noise, thresholds=[0.01], K=2000, is_uniform=True,
                   functions=None, function_names=None, vars=None):
    """
    Run weak form SINDy on time series data.

    Parameters:
    - t: array-like, shape (T,), time points
    - x: array-like, shape (T, n_features), state data
    - thresholds: list of floats, sparsity thresholds for SR3
    - K: int, number of test functions
    - is_uniform: bool, whether the grid is uniform
    - functions: list of callables, library functions
    - function_names: list of callables returning string names
    - vars: list of str, variable names for feature_names
    - noise: flag declaring noisy data. The choise o optimizer depends on this
    
    Returns:
    - list of tuples: (threshold, smodel, r2_score)
    """
    results = []

    if functions is None:
        functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
    if vars is None:
        vars = [f"x{i}" for i in range(x.shape[1])]

    # Proper symbolic names based on 'vars'
    function_names = [
        lambda x: x,
        lambda x: f"{x}^2",
        lambda x, y: f"{x}{y}"
    ]
    spatiotemporal_grid = t[:, None]

    ode_lib = WeakPDELibrary(
        library_functions=functions,
        function_names=function_names,
        spatiotemporal_grid=spatiotemporal_grid,
        is_uniform=is_uniform,
        K=K,
    )

    data_dot_integral = ode_lib.convert_u_dot_integral(x)

    for threshold in thresholds:
        optimizer = (
            STLSQ(threshold=threshold,normalize_columns=False) 
            if noise else 
            SR3(threshold=threshold, thresholder="l1", max_iter=1000, normalize_columns=False, tol=1e10)
        )
        smodel = SINDy(
            feature_library=ode_lib,
            optimizer=optimizer,
            feature_names=vars,
        )

        # 🔇 Suppress only the STLSQ warning about eliminated coefficients
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=UserWarning)
            smodel.fit(x, x_dot=data_dot_integral, t=t[1] - t[0], ensemble=True)
            x_dot_pred = smodel.predict(x)

        # ✅ Original R² score using weak-form data
        r2 = r2_score(data_dot_integral, x_dot_pred)
        r21 = r2_score(data_dot_integral[:,0], x_dot_pred[:,0])
        r22 = r2_score(data_dot_integral[:,1], x_dot_pred[:,1])

        results.append((threshold, smodel, r2))
        
        print(f"Threshold = {threshold}")
        print("Dynamic model:")
        smodel.print()
        # print(f"R² score: {r2:.5f}")
        print(f"R²({vars[0]}) score: {r21:.5f}")
        print(f"R²({vars[1]}) score: {r22:.5f}")
        print()

    return results