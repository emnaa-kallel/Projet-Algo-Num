"""
Visualisations pour les problèmes CoolingProblem et FlowProblem
"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.problems import CoolingProblem, FlowProblem
from src.visualization import Visualizer


# =============================================================
# 1. VISUALISATION COOLING PROBLEM
# =============================================================

def visualize_cooling_problem():
    """Génère les graphes pour le problème de refroidissement."""
    print("\n📊 Génération des graphes : COOLING PROBLEM")
    
    # Données
    t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    T_data = np.array([100, 85.5, 73.6, 63.9, 56.1, 49.9, 44.9, 40.8, 37.4, 34.6, 32.3])
    
    cooling = CoolingProblem(t_data, T_data, T_ambient=20.0, h_coeff=50.0)
    
    # Points fins pour tracé lisse
    t_fine = np.linspace(t_data[0], t_data[-1], 300)
    T_interp = cooling.temperature(t_fine, method='newton')
    
    # Modèle exponentiel optimal
    k_opt = cooling.estimate_k()
    T_model = cooling.exponential_model(t_fine, k_opt)
    
    # Visualisation
    viz = Visualizer(figsize=(12, 6))
    fig = viz.plot_cooling_analysis(t_data, T_data, t_fine, T_interp, k_opt, T_model)
    
    # Sauvegarde
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", "cooling_analysis.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graphe sauvegardé : {filepath}")
    
    return fig


# =============================================================
# 2. VISUALISATION FLOW PROBLEM
# =============================================================

def visualize_flow_problem():
    """Génère les graphes pour le problème d'écoulement."""
    print("\n📊 Génération des graphes : FLOW PROBLEM")
    
    # Données
    x_data = np.array([0, 1, 2, 3, 4, 5, 6])
    v_data = np.array([0.5, 0.8, 1.1, 1.3, 1.4, 1.3, 1.0])
    
    flow = FlowProblem(x_data, v_data)
    
    # Points fins pour tracé lisse
    x_fine = np.linspace(x_data[0], x_data[-1], 300)
    v_interp = flow.velocity(x_fine, method='newton')
    
    # Visualisation
    viz = Visualizer(figsize=(14, 5))
    fig = viz.plot_flow_analysis(x_data, v_data, x_fine, v_interp, flow.width_func)
    
    # Sauvegarde
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", "flow_analysis.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graphe sauvegardé : {filepath}")
    
    return fig


# =============================================================
# 3. VISUALISATION COMPARÉE (COOLING)
# =============================================================

def visualize_cooling_interpolations():
    """Compare les deux méthodes d'interpolation pour le cooling."""
    print("\n📊 Génération des graphes : COMPARAISON INTERPOLATIONS (COOLING)")
    
    # Données
    t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    T_data = np.array([100, 85.5, 73.6, 63.9, 56.1, 49.9, 44.9, 40.8, 37.4, 34.6, 32.3])
    
    cooling = CoolingProblem(t_data, T_data)
    
    # Points fins pour tracé
    t_fine = np.linspace(t_data[0], t_data[-1], 300)
    T_lag = cooling.temperature(t_fine, method='lagrange')
    T_new = cooling.temperature(t_fine, method='newton')
    
    # Visualisation
    viz = Visualizer(figsize=(12, 6))
    interpolators = {
        "Lagrange": T_lag,
        "Newton": T_new
    }
    fig = viz.plot_interpolation_comparison(t_data, T_data, interpolators, t_fine, 
                                           title="Comparaison des interpolations — Refroidissement")
    
    # Sauvegarde
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", "cooling_interpolations_comparison.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graphe sauvegardé : {filepath}")
    
    return fig


# =============================================================
# 4. VISUALISATION COMPARÉE (FLOW)
# =============================================================

def visualize_flow_interpolations():
    """Compare les deux méthodes d'interpolation pour le flow."""
    print("\n📊 Génération des graphes : COMPARAISON INTERPOLATIONS (FLOW)")
    
    # Données
    x_data = np.array([0, 1, 2, 3, 4, 5, 6])
    v_data = np.array([0.5, 0.8, 1.1, 1.3, 1.4, 1.3, 1.0])
    
    flow = FlowProblem(x_data, v_data)
    
    # Points fins pour tracé
    x_fine = np.linspace(x_data[0], x_data[-1], 300)
    v_lag = flow.velocity(x_fine, method='lagrange')
    v_new = flow.velocity(x_fine, method='newton')
    
    # Visualisation
    viz = Visualizer(figsize=(12, 6))
    interpolators = {
        "Lagrange": v_lag,
        "Newton": v_new
    }
    fig = viz.plot_interpolation_comparison(x_data, v_data, interpolators, x_fine,
                                           title="Comparaison des interpolations — Écoulement")
    
    # Sauvegarde
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", "flow_interpolations_comparison.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graphe sauvegardé : {filepath}")
    
    return fig


# =============================================================
# 5. VISUALISATION INTÉGRATION (CONVERGENCE)
# =============================================================

def visualize_cooling_heat_convergence():
    """Convergence de l'intégration pour la chaleur dissipée."""
    print("\n📊 Génération des graphes : CONVERGENCE CHALEUR (COOLING)")
    
    # Données
    t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    T_data = np.array([100, 85.5, 73.6, 63.9, 56.1, 49.9, 44.9, 40.8, 37.4, 34.6, 32.3])
    
    cooling = CoolingProblem(t_data, T_data)
    
    # Valeur de référence (adaptative avec haute précision)
    Q_ref = cooling.total_heat_loss(method='adaptive')
    
    # Convergence pour différentes valeurs de n
    n_values = [5, 10, 20, 50, 100, 200, 500]
    errors = {
        "Trapèze": [],
        "Simpson": [],
    }
    
    for n in n_values:
        Q_trap = cooling.total_heat_loss(method='trapeze', n=n)
        Q_simp = cooling.total_heat_loss(method='simpson', n=n)
        errors["Trapèze"].append(abs(Q_trap - Q_ref))
        errors["Simpson"].append(abs(Q_simp - Q_ref))
    
    # Visualisation
    viz = Visualizer(figsize=(10, 6))
    fig = viz.plot_convergence(
        n_values,
        errors,
        title="Convergence de l'intégration — Chaleur dissipée",
        xlabel="n (subdivisions)",
        ylabel="Erreur absolue |Q_n - Q_ref|"
    )
    
    # Sauvegarde
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", "cooling_convergence.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graphe sauvegardé : {filepath}")
    
    return fig


# =============================================================
# 6. VISUALISATION INTÉGRATION (DÉBIT)
# =============================================================

def visualize_flow_debit_convergence():
    """Convergence de l'intégration pour le débit total."""
    print("\n📊 Génération des graphes : CONVERGENCE DÉBIT (FLOW)")
    
    # Données
    x_data = np.array([0, 1, 2, 3, 4, 5, 6])
    v_data = np.array([0.5, 0.8, 1.1, 1.3, 1.4, 1.3, 1.0])
    
    flow = FlowProblem(x_data, v_data)
    
    # Valeur de référence (adaptative avec haute précision)
    D_ref = flow.total_flow_rate(method='adaptive')
    
    # Convergence pour différentes valeurs de n
    n_values = [5, 10, 20, 50, 100, 200, 500]
    errors = {
        "Trapèze": [],
        "Simpson": [],
    }
    
    for n in n_values:
        D_trap = flow.total_flow_rate(method='trapeze', n=n)
        D_simp = flow.total_flow_rate(method='simpson', n=n)
        errors["Trapèze"].append(abs(D_trap - D_ref))
        errors["Simpson"].append(abs(D_simp - D_ref))
    
    # Visualisation
    viz = Visualizer(figsize=(10, 6))
    fig = viz.plot_convergence(
        n_values,
        errors,
        title="Convergence de l'intégration — Débit volumique",
        xlabel="n (subdivisions)",
        ylabel="Erreur absolue |D_n - D_ref|"
    )
    
    # Sauvegarde
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", "flow_convergence.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✅ Graphe sauvegardé : {filepath}")
    
    return fig


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  VISUALISATIONS COMPLÈTES - COOLING ET FLOW PROBLEMS")
    print("=" * 70)
    
    # Graphes principaux
    visualize_cooling_problem()
    visualize_flow_problem()
    
    # Comparaisons d'interpolations
    visualize_cooling_interpolations()
    visualize_flow_interpolations()
    
    # Convergence de l'intégration
    visualize_cooling_heat_convergence()
    visualize_flow_debit_convergence()
    
    print("\n" + "=" * 70)
    print("  ✅ TOUTES LES VISUALISATIONS GÉNÉRÉES")
    print("  📁 Les graphes sont dans le dossier 'results/'")
    print("=" * 70)
