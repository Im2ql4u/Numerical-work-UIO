"""
Quantum gate definitions.

Provides single-qubit gates (Pauli X, Y, Z, Hadamard, Phase, Rx, Ry, Rz, U3)
and multi-qubit gates (CNOT, controlled-Z).
"""

import numpy as np

# =============================================================================
# Pauli matrices
# =============================================================================
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# =============================================================================
# Standard single-qubit gates
# =============================================================================
H_gate = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S_gate = np.array([[1, 0], [0, 1j]], dtype=complex)
T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def Rx(theta: float) -> np.ndarray:
    """Rotation about X-axis by angle theta."""
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    """Rotation about Y-axis by angle theta."""
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)


def Rz(phi: float) -> np.ndarray:
    """Rotation about Z-axis by angle phi."""
    return np.array([
        [np.exp(-1j * phi / 2), 0],
        [0, np.exp(1j * phi / 2)]
    ], dtype=complex)


def U3(theta: float, phi: float, lam: float) -> np.ndarray:
    """
    General single-qubit unitary U3(theta, phi, lambda).
    
    U3|0⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    """
    return np.array([
        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2),
         np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
    ], dtype=complex)


# =============================================================================
# Multi-qubit gates
# =============================================================================
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)
