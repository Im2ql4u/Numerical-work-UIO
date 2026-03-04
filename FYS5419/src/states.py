"""
Quantum state definitions.

Provides computational basis states and Bell state construction.
"""

import numpy as np
from .gates import H_gate, CNOT, I, X

# =============================================================================
# Computational basis states
# =============================================================================
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)


def ket(label: str) -> np.ndarray:
    """
    Return a multi-qubit computational basis state from a binary string.
    
    Parameters
    ----------
    label : str
        Binary string, e.g. '00', '01', '10', '11'.
    
    Returns
    -------
    np.ndarray
        The state vector.
    """
    state = np.array([1], dtype=complex)
    for bit in label:
        if bit == '0':
            state = np.kron(state, ket_0)
        elif bit == '1':
            state = np.kron(state, ket_1)
        else:
            raise ValueError(f"Invalid bit: {bit}")
    return state


# =============================================================================
# Bell states
# =============================================================================
def bell_states() -> dict:
    """
    Construct all four Bell states.
    
    Returns
    -------
    dict
        Mapping from name to state vector:
        - 'Φ+' = (|00⟩ + |11⟩) / √2
        - 'Φ-' = (|00⟩ - |11⟩) / √2
        - 'Ψ+' = (|01⟩ + |10⟩) / √2
        - 'Ψ-' = (|01⟩ - |10⟩) / √2
    """
    s00 = ket('00')
    s01 = ket('01')
    s10 = ket('10')
    s11 = ket('11')
    
    return {
        'Φ+': (s00 + s11) / np.sqrt(2),
        'Φ-': (s00 - s11) / np.sqrt(2),
        'Ψ+': (s01 + s10) / np.sqrt(2),
        'Ψ-': (s01 - s10) / np.sqrt(2),
    }


def create_bell_state_circuit(initial_state: np.ndarray) -> np.ndarray:
    """
    Apply Hadamard on qubit 0 then CNOT to a two-qubit initial state.
    This is the standard Bell-state creation circuit.
    
    Parameters
    ----------
    initial_state : np.ndarray
        A two-qubit state vector (length 4).
    
    Returns
    -------
    np.ndarray
        The resulting state after H⊗I followed by CNOT.
    """
    H_on_first = np.kron(H_gate, I)
    return CNOT @ (H_on_first @ initial_state)


def apply_bell_measurement_circuit(state: np.ndarray) -> np.ndarray:
    """
    Apply the inverse Bell circuit (CNOT then H⊗I) to decode a Bell state
    back into the computational basis.
    
    Parameters
    ----------
    state : np.ndarray
        A two-qubit state vector (length 4).
    
    Returns
    -------
    np.ndarray
        The decoded state.
    """
    H_on_first = np.kron(H_gate, I)
    return H_on_first @ (CNOT @ state)
