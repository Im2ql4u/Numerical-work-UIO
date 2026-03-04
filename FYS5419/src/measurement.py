"""
Measurement simulation.

Provides shot-based measurement of quantum states in the computational basis
and in arbitrary Pauli bases.
"""

import numpy as np
from .gates import I, H_gate, S_gate


def measure_computational_basis(state: np.ndarray, shots: int = 10000) -> dict:
    """
    Simulate shots measurements of a quantum state in the computational basis.
    
    Parameters
    ----------
    state : np.ndarray
        State vector of n qubits (length 2^n).
    shots : int
        Number of measurement shots.
    
    Returns
    -------
    dict
        Mapping from basis state index to measured count.
    """
    n_states = len(state)
    n_qubits = int(np.log2(n_states))
    probs = np.abs(state) ** 2
    probs /= probs.sum()  # normalise for safety
    
    outcomes = np.random.choice(n_states, size=shots, p=probs)
    counts = {}
    for i in range(n_states):
        label = format(i, f'0{n_qubits}b')
        c = np.count_nonzero(outcomes == i)
        if c > 0:
            counts[label] = c
    return counts


def measure_single_qubit_z(state: np.ndarray, qubit: int,
                            n_qubits: int, shots: int = 10000) -> float:
    """
    Measure expectation value of Z on a single qubit via shot-based simulation.
    
    Parameters
    ----------
    state : np.ndarray
        State vector.
    qubit : int
        Which qubit to measure (0-indexed, qubit 0 is leftmost / most significant).
    n_qubits : int
        Total number of qubits.
    shots : int
        Number of measurement shots.
    
    Returns
    -------
    float
        Estimated ⟨Z_qubit⟩.
    """
    probs = np.abs(state) ** 2
    probs /= probs.sum()
    outcomes = np.random.choice(len(state), size=shots, p=probs)
    
    expectation = 0.0
    for outcome in outcomes:
        # Extract bit for the target qubit
        bit = (outcome >> (n_qubits - 1 - qubit)) & 1
        expectation += (-1) ** bit
    return expectation / shots


def _rotation_for_pauli(pauli_char: str) -> np.ndarray:
    """Return the basis rotation matrix for measuring in a given Pauli basis."""
    if pauli_char == 'Z' or pauli_char == 'I':
        return I
    elif pauli_char == 'X':
        return H_gate
    elif pauli_char == 'Y':
        Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
        return H_gate @ Sdg
    else:
        raise ValueError(f"Unknown Pauli: {pauli_char}")


def measure_pauli_string(state: np.ndarray, pauli_str: str,
                          shots: int = 10000) -> float:
    """
    Measure the expectation value of a Pauli string operator via shot-based
    simulation.
    
    Parameters
    ----------
    state : np.ndarray
        State vector of n qubits.
    pauli_str : str
        String of Pauli operators, e.g. 'XZ', 'IY'. Length must equal n_qubits.
        Convention: pauli_str[0] acts on qubit 0 (most significant bit).
    shots : int
        Number of measurement shots.
    
    Returns
    -------
    float
        Estimated expectation value ⟨pauli_str⟩.
    """
    n_qubits = len(pauli_str)
    
    # Build the full rotation matrix
    rotation = np.array([1], dtype=complex)
    for char in pauli_str:
        rotation = np.kron(rotation, _rotation_for_pauli(char))
    
    rotated_state = rotation @ state
    probs = np.abs(rotated_state) ** 2
    probs /= probs.sum()
    
    outcomes = np.random.choice(len(state), size=shots, p=probs)
    
    expectation = 0.0
    for outcome in outcomes:
        # Eigenvalue is (-1)^(number of 1s in the bits corresponding to non-I Paulis)
        parity = 0
        for q, char in enumerate(pauli_str):
            if char != 'I':
                bit = (outcome >> (n_qubits - 1 - q)) & 1
                parity += bit
        expectation += (-1) ** parity
    
    return expectation / shots


def expectation_exact(state: np.ndarray, operator: np.ndarray) -> float:
    """Compute exact expectation value ⟨ψ|O|ψ⟩."""
    return np.real(np.vdot(state, operator @ state))
