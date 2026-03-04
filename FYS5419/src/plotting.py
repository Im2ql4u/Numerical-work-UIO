"""
Plotting utilities for consistent, well-labeled figures.
Uses the project Thesis_style.mplstyle for visual identity,
scaled for notebook/report use.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def setup_plotting(style_path=None):
    """
    Load thesis-style aesthetics (colors, fonts, grid, background)
    with notebook-appropriate sizing.

    Parameters
    ----------
    style_path : str, optional
        Path to .mplstyle file.  Auto-detected if None.
    """
    if style_path is None:
        # Look relative to this file's parent (project root)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        style_path = os.path.join(root, 'Thesis_style.mplstyle')

    if os.path.isfile(style_path):
        plt.style.use(style_path)

    # Override sizes that are too large for notebook / PDF figures
    plt.rcParams.update({
        'figure.figsize': (12, 7),
        'figure.dpi': 150,
        'font.size': 14,
        'axes.titlesize': 17,
        'axes.labelsize': 15,
        'axes.titleweight': 'semibold',
        'axes.labelweight': 'medium',
        'axes.titley': 1.02,
        'legend.fontsize': 13,
        'legend.framealpha': 0.9,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'xtick.major.size': 7,
        'ytick.major.size': 7,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
        'lines.linewidth': 2.5,
        'lines.markersize': 6,
        'figure.titlesize': 19,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })


def plot_eigenvalues_vs_param(param_values, all_eigenvalues, xlabel, title,
                               save_path=None):
    """
    Plot all eigenvalues as a function of a sweep parameter.
    
    Parameters
    ----------
    param_values : array-like
        Sweep parameter values (x-axis).
    all_eigenvalues : list of arrays
        List where each element is the array of eigenvalues at that parameter value.
    xlabel : str
        Label for x-axis.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    """
    fig, ax = plt.subplots()
    n_eigs = len(all_eigenvalues[0])
    eig_arrays = np.array(all_eigenvalues)
    
    for i in range(n_eigs):
        ax.plot(param_values, eig_arrays[:, i], label=f'$E_{i}$')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Energy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_vqe_comparison(param_values, exact_energies, vqe_energies,
                         xlabel, title, save_path=None):
    """
    Plot exact vs VQE ground state energies.
    """
    fig, ax = plt.subplots()
    ax.plot(param_values, exact_energies, 'o-', label='Exact', markersize=4, alpha=0.8)
    ax.plot(param_values, vqe_energies, 's--', label='VQE', markersize=4, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Ground State Energy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_entropy(param_values, entropies, xlabel, title, save_path=None):
    """
    Plot von Neumann entropy as a function of a parameter.
    """
    fig, ax = plt.subplots()
    ax.plot(param_values, entropies, 'b-', linewidth=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Von Neumann Entropy (bits)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_combined_vqe_lipkin(results_j1, results_j2, save_path=None):
    """
    Plot VQE vs exact for both J=1 and J=2 Lipkin model on one figure.
    """
    fig, ax = plt.subplots()
    
    ax.plot(results_j1['param_values'], results_j1['exact_energies'],
            'o-', label='Exact $J=1$', markersize=4, color='C0')
    ax.plot(results_j1['param_values'], results_j1['vqe_energies'],
            's--', label='VQE $J=1$', markersize=4, color='C0', alpha=0.6)
    
    ax.plot(results_j2['param_values'], results_j2['exact_energies'],
            'o-', label='Exact $J=2$', markersize=4, color='C1')
    ax.plot(results_j2['param_values'], results_j2['vqe_energies'],
            's--', label='VQE $J=2$', markersize=4, color='C1', alpha=0.6)
    
    ax.set_xlabel('Interaction Strength $V$')
    ax.set_ylabel('Ground State Energy')
    ax.set_title('Lipkin Model: VQE vs Exact Diagonalization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
