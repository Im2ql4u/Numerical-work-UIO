"""
Density matrix operations and von Neumann entropy.
"""

import numpy as np


def density_matrix(state: np.ndarray) -> np.ndarray:
    """Construct density matrix ρ = |ψ⟩⟨ψ| from a pure state vector."""
    return np.outer(state, np.conj(state))


def partial_trace_B(rho: np.ndarray, dim_A: int, dim_B: int) -> np.ndarray:
    """
    Compute the partial trace over subsystem B: ρ_A = Tr_B(ρ).
    
    Parameters
    ----------
    rho : np.ndarray
        Density matrix of the composite system (dim_A * dim_B) x (dim_A * dim_B).
    dim_A : int
        Dimension of subsystem A.
    dim_B : int
        Dimension of subsystem B.
    
    Returns
    -------
    np.ndarray
        Reduced density matrix ρ_A of shape (dim_A, dim_A).
    """
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho_reshaped, axis1=1, axis2=3)


def partial_trace_A(rho: np.ndarray, dim_A: int, dim_B: int) -> np.ndarray:
    """
    Compute the partial trace over subsystem A: ρ_B = Tr_A(ρ).
    
    Parameters
    ----------
    rho : np.ndarray
        Density matrix of the composite system (dim_A * dim_B) x (dim_A * dim_B).
    dim_A : int
        Dimension of subsystem A.
    dim_B : int
        Dimension of subsystem B.
    
    Returns
    -------
    np.ndarray
        Reduced density matrix ρ_B of shape (dim_B, dim_B).
    """
    rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
    return np.trace(rho_reshaped, axis1=0, axis2=2)


def von_neumann_entropy(rho: np.ndarray, base: int = 2) -> float:
    """
    Compute the von Neumann entropy S(ρ) = -Tr(ρ log ρ).
    
    Parameters
    ----------
    rho : np.ndarray
        Density matrix.
    base : int
        Logarithm base (2 for bits, e for nats).
    
    Returns
    -------
    float
        The von Neumann entropy.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.real(eigenvalues)
    # Filter out zero/negative eigenvalues (numerical noise)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if base == 2:
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    else:
        return -np.sum(eigenvalues * np.log(eigenvalues))
