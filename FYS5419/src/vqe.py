"""
Variational Quantum Eigensolver (VQE) implementation.

Provides ansatz circuits and the VQE optimization loop for:
- One-qubit systems (Part c)
- Two-qubit systems (Part e)
- Lipkin model J=1 and J=2 (Part g)
"""

import numpy as np
from scipy.optimize import minimize
from .gates import Ry, Rz, CNOT, I
from .measurement import expectation_exact


# =============================================================================
# Ansatz circuits
# =============================================================================
def one_qubit_ansatz(params: np.ndarray) -> np.ndarray:
    """
    One-qubit variational ansatz: |ψ(θ,φ)⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.
    
    This is equivalent to U3(θ, φ, 0)|0⟩, or Rz(φ) Ry(θ)|0⟩.
    
    Parameters
    ----------
    params : array-like
        [theta, phi].
    
    Returns
    -------
    np.ndarray
        State vector of length 2.
    """
    theta, phi = params
    return np.array([
        np.cos(theta / 2),
        np.exp(1j * phi) * np.sin(theta / 2)
    ], dtype=complex)


def two_qubit_ansatz(params: np.ndarray) -> np.ndarray:
    """
    Two-qubit variational ansatz with entanglement.
    
    Circuit:
    |0⟩ —[Ry(θ₁)]—●—[Ry(θ₃)]—
    |0⟩ —[Ry(θ₂)]—X—[Ry(θ₄)]—
    
    Parameters
    ----------
    params : array-like
        [theta1, theta2, theta3, theta4].
    
    Returns
    -------
    np.ndarray
        State vector of length 4.
    """
    theta1, theta2, theta3, theta4 = params
    state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    
    # Layer 1: Ry rotations
    U1 = np.kron(Ry(theta1), Ry(theta2))
    state = U1 @ state
    
    # Entangling: CNOT
    state = CNOT @ state
    
    # Layer 2: Ry rotations
    U2 = np.kron(Ry(theta3), Ry(theta4))
    state = U2 @ state
    
    return state


def lipkin_j1_ansatz(params: np.ndarray) -> np.ndarray:
    """
    Ansatz for J=1 Lipkin model (4-parameter hyperspherical).
    
    Produces a general 3-component state vector:
    |ψ⟩ = [cos θ₁,
            sin θ₁ cos θ₂ e^{iφ₁},
            sin θ₁ sin θ₂ e^{iφ₂}]  / N
    
    Parameters
    ----------
    params : array-like
        [theta1, theta2, phi1, phi2].
    
    Returns
    -------
    np.ndarray
        Normalized state vector of length 3.
    """
    theta1, theta2, phi1, phi2 = params
    state = np.array([
        np.cos(theta1),
        np.sin(theta1) * np.cos(theta2) * np.exp(1j * phi1),
        np.sin(theta1) * np.sin(theta2) * np.exp(1j * phi2),
    ], dtype=complex)
    norm = np.linalg.norm(state)
    if norm > 1e-10:
        state = state / norm
    return state


def lipkin_j2_ansatz(params: np.ndarray) -> np.ndarray:
    """
    Ansatz for J=2 Lipkin model (8-parameter).
    
    Produces a general 5-dimensional state vector using a hyperspherical
    parametrization:
    
    |ψ⟩ = [cos θ₁,
            sin θ₁ cos θ₂ e^{iφ₁},
            sin θ₁ sin θ₂ cos θ₃ e^{iφ₂},
            sin θ₁ sin θ₂ sin θ₃ cos θ₄ e^{iφ₃},
            sin θ₁ sin θ₂ sin θ₃ sin θ₄ e^{iφ₄}]  / N
    
    Parameters
    ----------
    params : array-like
        [theta1, theta2, theta3, theta4, phi1, phi2, phi3, phi4].
    
    Returns
    -------
    np.ndarray
        Normalized state vector of length 5.
    """
    theta1, theta2, theta3, theta4, phi1, phi2, phi3, phi4 = params
    
    state = np.array([
        np.cos(theta1),
        np.sin(theta1) * np.cos(theta2) * np.exp(1j * phi1),
        np.sin(theta1) * np.sin(theta2) * np.cos(theta3) * np.exp(1j * phi2),
        np.sin(theta1) * np.sin(theta2) * np.sin(theta3) * np.cos(theta4) * np.exp(1j * phi3),
        np.sin(theta1) * np.sin(theta2) * np.sin(theta3) * np.sin(theta4) * np.exp(1j * phi4),
    ], dtype=complex)
    
    norm = np.linalg.norm(state)
    if norm > 1e-10:
        state = state / norm
    return state


# =============================================================================
# VQE optimizer
# =============================================================================
def run_vqe(hamiltonian: np.ndarray, ansatz_fn, n_params: int,
            n_restarts: int = 10, method: str = 'COBYLA',
            maxiter: int = 1000) -> dict:
    """
    Run the Variational Quantum Eigensolver.
    
    Parameters
    ----------
    hamiltonian : np.ndarray
        The Hamiltonian matrix.
    ansatz_fn : callable
        Function that takes parameter array and returns a state vector.
    n_params : int
        Number of variational parameters.
    n_restarts : int
        Number of random restarts for the optimizer.
    method : str
        Optimization method for scipy.optimize.minimize.
    maxiter : int
        Maximum iterations per optimization run.
    
    Returns
    -------
    dict
        'energy': best energy found,
        'params': optimal parameters,
        'state': optimal state vector.
    """
    best_energy = np.inf
    best_params = None
    
    for _ in range(n_restarts):
        params0 = np.random.uniform(0, 2 * np.pi, n_params)
        
        def cost(p):
            psi = ansatz_fn(p)
            return expectation_exact(psi, hamiltonian)
        
        result = minimize(cost, params0, method=method,
                          options={'maxiter': maxiter})
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x
    
    best_state = ansatz_fn(best_params)
    
    return {
        'energy': best_energy,
        'params': best_params,
        'state': best_state,
    }


def vqe_sweep(build_hamiltonian_fn, ansatz_fn, n_params: int,
              param_values: np.ndarray, n_restarts: int = 5,
              method: str = 'COBYLA', maxiter: int = 1000,
              verbose: bool = False) -> dict:
    """
    Sweep VQE over a range of parameter values (e.g., λ or V).
    
    Parameters
    ----------
    build_hamiltonian_fn : callable
        Function that takes a single parameter value and returns a Hamiltonian matrix.
    ansatz_fn : callable
        Ansatz function.
    n_params : int
        Number of variational parameters.
    param_values : np.ndarray
        Array of parameter values to sweep over.
    n_restarts : int
        Number of random restarts.
    method : str
        Optimization method.
    maxiter : int
        Maximum iterations.
    verbose : bool
        Print progress.
    
    Returns
    -------
    dict
        'param_values': the swept values,
        'exact_energies': exact ground state energies,
        'vqe_energies': VQE ground state energies,
        'all_exact_eigenvalues': list of all eigenvalue arrays.
    """
    exact_energies = []
    vqe_energies = []
    all_eigenvalues = []
    
    for i, val in enumerate(param_values):
        H = build_hamiltonian_fn(val)
        
        # Exact diagonalization
        eigvals = np.linalg.eigvalsh(H)
        all_eigenvalues.append(eigvals)
        exact_energies.append(eigvals[0])
        
        # VQE
        result = run_vqe(H, ansatz_fn, n_params, n_restarts, method, maxiter)
        vqe_energies.append(result['energy'])
        
        if verbose:
            print(f"  [{i+1}/{len(param_values)}] val={val:.3f}  "
                  f"exact={eigvals[0]:.6f}  vqe={result['energy']:.6f}")
    
    return {
        'param_values': param_values,
        'exact_energies': np.array(exact_energies),
        'vqe_energies': np.array(vqe_energies),
        'all_exact_eigenvalues': all_eigenvalues,
    }
