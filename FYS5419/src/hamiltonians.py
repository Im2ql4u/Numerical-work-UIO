"""
Hamiltonian construction for all parts of the project.

Part b: One-qubit λ-dependent Hamiltonian H = H₀ + λH_I
Part d: Two-qubit Hamiltonian with Hx, Hz interaction
Part f: Lipkin model for J=1 (3×3) and J=2 (5×5)
"""

import numpy as np
from .gates import I, X, Y, Z


# =============================================================================
# Part b: One-qubit Hamiltonian
# =============================================================================
def one_qubit_hamiltonian(E1: float = 0.0, E2: float = 4.0,
                          V11: float = 3.0, V22: float = -3.0,
                          V12: float = 0.2, lam: float = 1.0) -> np.ndarray:
    """
    Construct the one-qubit Hamiltonian H = H₀ + λ H_I.
    
    H₀ = ε I + Ω σ_z,  where ε = (E₁+E₂)/2, Ω = (E₁-E₂)/2
    H_I = c I + ω_z σ_z + ω_x σ_x,  where c = (V₁₁+V₂₂)/2, ω_z = (V₁₁-V₂₂)/2, ω_x = V₁₂
    
    Parameters
    ----------
    E1, E2 : float
        Non-interacting energies.
    V11, V22, V12 : float
        Interaction matrix elements (V21 = V12 for symmetric H).
    lam : float
        Interaction strength parameter λ ∈ [0, 1].
    
    Returns
    -------
    np.ndarray
        2×2 Hamiltonian matrix.
    """
    eps = (E1 + E2) / 2.0
    omega = (E1 - E2) / 2.0
    c = (V11 + V22) / 2.0
    wz = (V11 - V22) / 2.0
    wx = V12
    
    H0 = eps * I + omega * Z
    HI = c * I + wz * Z + wx * X
    return H0 + lam * HI


def one_qubit_pauli_coefficients(E1: float = 0.0, E2: float = 4.0,
                                  V11: float = 3.0, V22: float = -3.0,
                                  V12: float = 0.2, lam: float = 1.0) -> dict:
    """
    Return the Pauli decomposition coefficients of the one-qubit Hamiltonian.
    
    H = a_I · I + a_x · σ_x + a_z · σ_z
    
    Returns
    -------
    dict
        {'I': a_I, 'X': a_x, 'Z': a_z}
    """
    eps = (E1 + E2) / 2.0
    omega = (E1 - E2) / 2.0
    c = (V11 + V22) / 2.0
    wz = (V11 - V22) / 2.0
    wx = V12
    
    return {
        'I': eps + lam * c,
        'X': lam * wx,
        'Z': omega + lam * wz,
    }


# =============================================================================
# Part d: Two-qubit Hamiltonian
# =============================================================================
def two_qubit_hamiltonian(Hx: float = 2.0, Hz: float = 3.0,
                          energies: np.ndarray = None,
                          lam: float = 1.0) -> np.ndarray:
    """
    Construct the two-qubit Hamiltonian:
    
    H = H₀ + λ H_I, where
    H₀ = diag(ε₀₀, ε₁₀, ε₀₁, ε₁₁)
    H_I = Hx (σ_x ⊗ σ_x) + Hz (σ_z ⊗ σ_z)
    
    The full matrix is:
    ⎡ ε₀₀ + Hz    0       0      Hx  ⎤
    ⎢  0     ε₁₀ - Hz    Hx      0   ⎥
    ⎢  0       Hx     ε₀₁ - Hz   0   ⎥
    ⎣ Hx       0       0     ε₁₁ + Hz ⎦
    
    Parameters
    ----------
    Hx, Hz : float
        Interaction strengths.
    energies : np.ndarray
        Non-interacting energies [ε₀₀, ε₁₀, ε₀₁, ε₁₁].
    lam : float
        Interaction strength parameter λ.
    
    Returns
    -------
    np.ndarray
        4×4 Hamiltonian matrix.
    """
    if energies is None:
        energies = np.array([0.0, 2.5, 6.5, 7.0])
    
    H0 = np.diag(energies)
    HI = Hx * np.kron(X, X) + Hz * np.kron(Z, Z)
    return H0 + lam * HI


def two_qubit_pauli_decomposition(Hx: float = 2.0, Hz: float = 3.0,
                                   energies: np.ndarray = None,
                                   lam: float = 1.0) -> list:
    """
    Decompose the two-qubit Hamiltonian into Pauli terms.
    
    Returns
    -------
    list of (float, str)
        Each entry is (coefficient, pauli_string), e.g. (2.5, 'IZ').
    """
    H = two_qubit_hamiltonian(Hx, Hz, energies, lam)
    return pauli_decompose_matrix(H, n_qubits=2)


# =============================================================================
# Part f: Lipkin model Hamiltonians
# =============================================================================
def lipkin_j1_matrix(eps: float, V: float) -> np.ndarray:
    """
    Construct the J=1 Lipkin Hamiltonian (3×3 matrix) with W=0.
    
    Using quasi-spin operators with J=1 (N=2 particles), the matrix is:
    
    H = ⎡ -ε   0   -V ⎤
        ⎢  0   0    0 ⎥
        ⎣ -V   0    ε ⎦
    
    Basis: |J=1, M=-1⟩, |J=1, M=0⟩, |J=1, M=+1⟩
    
    Parameters
    ----------
    eps : float
        Single-particle energy ε.
    V : float
        Interaction strength.
    
    Returns
    -------
    np.ndarray
        3×3 Hamiltonian matrix.
    """
    return np.array([
        [-eps,  0,  -V],
        [  0,   0,   0],
        [ -V,   0, eps]
    ], dtype=complex)


def lipkin_j2_matrix(eps: float, V: float, W: float = 0.0) -> np.ndarray:
    """
    Construct the J=2 Lipkin Hamiltonian (5×5 matrix).
    
    H = ⎡ -2ε        0       √6·V      0        0    ⎤
        ⎢   0     -ε+3W       0       3V        0    ⎥
        ⎢ √6·V      0        4W        0      √6·V  ⎥
        ⎢   0       3V        0       ε+3W       0    ⎥
        ⎣   0        0       √6·V      0       2ε    ⎦
    
    Parameters
    ----------
    eps : float
        Single-particle energy ε.
    V : float
        Two-body interaction strength.
    W : float
        Additional interaction (set to 0 by default).
    
    Returns
    -------
    np.ndarray
        5×5 Hamiltonian matrix.
    """
    sq6V = np.sqrt(6) * V
    return np.array([
        [-2*eps,     0,    sq6V,      0,      0],
        [    0, -eps+3*W,    0,     3*V,      0],
        [ sq6V,      0,   4*W,       0,   sq6V],
        [    0,     3*V,    0,   eps+3*W,     0],
        [    0,      0,   sq6V,      0,   2*eps]
    ], dtype=complex)


def lipkin_j1_pauli(eps: float, V: float) -> np.ndarray:
    """
    Build the J=1 Lipkin Hamiltonian embedded in 2-qubit (4×4) space.
    
    The physical 3×3 matrix is placed in the top-left block with the
    encoding: |M=-1⟩→|00⟩, |M=0⟩→|01⟩, |M=+1⟩→|10⟩.
    The |11⟩ state is unphysical and decoupled (zero row/column).
    
    Returns
    -------
    np.ndarray
        3×3 Hamiltonian (same as lipkin_j1_matrix).
    """
    return lipkin_j1_matrix(eps, V)


def lipkin_j2_pauli(eps: float, V: float) -> np.ndarray:
    """
    Build the J=2 Lipkin Hamiltonian (W=0).
    
    The 5×5 matrix decomposes into parity blocks:
    - Even block (M=-2,0,+2): 3×3
    - Odd block (M=-1,+1): 2×2
    
    For qubit encoding, the even block embeds in a 2-qubit (4×4) space
    and the odd block is a single-qubit (2×2) operator:
      H_odd = 3V σ_x - ε σ_z
    
    Returns
    -------
    np.ndarray
        5×5 Hamiltonian matrix (same as lipkin_j2_matrix with W=0).
    """
    return lipkin_j2_matrix(eps, V, W=0.0)


def lipkin_pauli_decomposition(J: int, eps: float, V: float) -> list:
    """
    Get the Pauli decomposition of a Lipkin Hamiltonian by embedding
    the physical matrix in qubit space.
    
    For J=1 (3×3 → 4×4, 2 qubits):
        Encoding: |M=-1⟩→|00⟩, |M=0⟩→|01⟩, |M=+1⟩→|10⟩, |11⟩ unphysical.
    
    For J=2 (5×5 → 8×8, 3 qubits):
        Encoding: states mapped to first 5 computational basis states.
    
    Parameters
    ----------
    J : int
        Total spin (1 or 2).
    eps : float
        Single-particle energy.
    V : float
        Interaction strength.
    
    Returns
    -------
    list of (float, str)
        Pauli decomposition terms (coefficient, pauli_string).
    """
    if J == 1:
        H_phys = lipkin_j1_matrix(eps, V)
        n_qubits = 2
        dim = 4
    elif J == 2:
        H_phys = lipkin_j2_matrix(eps, V, W=0.0)
        n_qubits = 3
        dim = 8
    else:
        raise ValueError("J must be 1 or 2.")
    
    # Embed physical matrix in qubit space
    H_embed = np.zeros((dim, dim), dtype=complex)
    d = H_phys.shape[0]
    H_embed[:d, :d] = H_phys
    
    return pauli_decompose_matrix(H_embed, n_qubits=n_qubits)


# =============================================================================
# Utility: General Pauli decomposition
# =============================================================================
def pauli_decompose_matrix(H: np.ndarray, n_qubits: int = None) -> list:
    """
    Decompose a Hermitian matrix into a sum of Pauli strings.
    
    H = Σ_i c_i P_i,  where P_i ∈ {I, X, Y, Z}^⊗n
    
    Parameters
    ----------
    H : np.ndarray
        Hermitian matrix of size 2^n × 2^n.
    n_qubits : int, optional
        Number of qubits. Inferred from H if not given.
    
    Returns
    -------
    list of (float, str)
        Non-zero (coefficient, pauli_string) pairs.
    """
    if n_qubits is None:
        n_qubits = int(np.log2(H.shape[0]))
    
    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    labels = list(paulis.keys())
    dim = 2 ** n_qubits
    
    terms = []
    
    def _recurse(depth, current_label, current_mat):
        if depth == n_qubits:
            coeff = np.trace(H @ current_mat).real / dim
            if abs(coeff) > 1e-10:
                terms.append((coeff, current_label))
            return
        for label, mat in paulis.items():
            _recurse(depth + 1, current_label + label,
                     np.kron(current_mat, mat) if current_mat.size > 1
                     else mat)
    
    for label, mat in paulis.items():
        _recurse(1, label, mat)
    
    return terms
