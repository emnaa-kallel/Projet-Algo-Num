"""
=============================================================
  Projet Analyse Numérique — main.py
  Interpolation + Intégration + Problèmes réels
=============================================================
"""
import os
import sys
import math
import numpy as np
import pandas as pd

from src.interpolation import PolynomialInterpolation
from src.integration import NewtonCotes, AdaptiveIntegration
from src.problems import CoolingProblem, FlowProblem
from src.visualization import Visualizer


# =============================================================
# 1. TEST SIMPLE — f(x) = eˣ sur [0,1]  (valeur exacte = e-1)
# =============================================================

def test_simple():
    """Test simple d'interpolation et intégration."""
    print("\n" + "=" * 70)
    print("  TEST SIMPLE — f(x) = eˣ, ∫₀¹ eˣ dx = e-1")
    print("=" * 70)

    x_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_pts = np.exp(x_pts)
    interp = PolynomialInterpolation(x_pts, y_pts)

    print("\n  📌 Interpolation :")
    for x_q in [0.3, 0.6, 0.9]:
        exact = np.exp(x_q)
        lag = interp.evaluate(x_q, 'lagrange')
        new = interp.evaluate(x_q, 'newton')
        print(f"    eˣ({x_q}) exact={exact:.6f} | Lagrange={lag:.6f} | Newton={new:.6f}")

    exact_int = np.e - 1
    f = np.exp
    ai = AdaptiveIntegration()
    
    print(f"\n  📌 Intégration ∫₀¹ eˣ dx (exact = {exact_int:.8f})")
    for name, val in [
        ("Rectangle  ", NewtonCotes.rectangle(f, 0, 1, 100)),
        ("Trapèze    ", NewtonCotes.trapezoidal(f, 0, 1, 100)),
        ("Simpson    ", NewtonCotes.simpson(f, 0, 1, 100)),
        ("Adaptatif  ", ai.adaptive_simpson(f, 0, 1)),
    ]:
        error = abs(val - exact_int)
        print(f"    {name}: {val:.8f}  (erreur = {error:.2e})")


# =============================================================
# 2. PHÉNOMÈNE DE RUNGE + TCHEBYCHEV 
# =============================================================

def test_runge():
    """Test du phénomène de Runge — Nœuds équidistants vs Tchebychev.
    
    ⚠️ Nécessite demonstrate_runge() du module interpolation.
    """
    print("\n" + "=" * 70)
    print("  PHÉNOMÈNE DE RUNGE — équidistants vs Tchebychev")
    print("=" * 70)
    
    try:
        from src.interpolation import demonstrate_runge
        
        data = demonstrate_runge([5, 10, 15, 20])
        print(f"\n  {'n':>4}  {'Erreur équidist.':>18}  {'Erreur Tchebychev':>18}")
        for n in [5, 10, 15, 20]:
            e_u = data["uniform"][n]["max_error"]
            e_c = data["chebyshev"][n]["max_error"]
            print(f"  {n:>4}  {e_u:>18.6f}  {e_c:>18.6f}")
        return data
    except ImportError:
        print("  ⚠️  demonstrate_runge() non disponible")
        return None


# =============================================================
# 3. ANALYSE DE CONVERGENCE — ∫₀¹ eˣ dx 
# =============================================================

def test_convergence():
    """Analyse de convergence pour ∫₀¹ eˣ dx.
    
    ⚠️ Nécessite convergence_analysis() du module integration.
    """
    print("\n" + "=" * 70)
    print("  CONVERGENCE — ∫₀¹ eˣ dx (exact = e-1)")
    print("=" * 70)
    
    try:
        from src.integration import convergence_analysis
        
        exact = np.e - 1
        errors = convergence_analysis(np.exp, 0, 1, exact,
                                      n_list=[2, 4, 8, 16, 32, 64, 128, 256])
        print(f"\n  {'n':>5}  {'Rectangle':>12}  {'Trapèze':>12}  {'Simpson':>12}")
        for i, n in enumerate(errors["n"]):
            r = errors["rectangle"][i]
            t = errors["trapezoidal"][i]
            s = errors["simpson"][i]
            print(f"  {n:>5}  {r:>12.2e}  {t:>12.2e}  {s:>12.2e}")
        return errors
    except ImportError:
        print("  ⚠️  convergence_analysis() non disponible")
        return None


# =============================================================
# 4. PROBLÈME DE REFROIDISSEMENT
# =============================================================

def test_cooling():
    """Test du problème de refroidissement avec données CSV."""
    print("\n" + "=" * 70)
    print("  PROBLÈME DE REFROIDISSEMENT")
    print("=" * 70)
    
    csv_path = "data/cooling.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        cp = CoolingProblem(df["t"].values, df["T"].values, T_ambient=20.0, h_coeff=50.0)
        cp.report()
        return cp
    else:
        print(f"\n  ⚠️  Fichier {csv_path} introuvable.")
        print("  Utilisation de données synthétiques...\n")
        
        # Données synthétiques
        t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        T_data = np.array([100, 85.5, 73.6, 63.9, 56.1, 49.9, 44.9, 40.8, 37.4, 34.6, 32.3])
        cp = CoolingProblem(t_data, T_data, T_ambient=20.0, h_coeff=50.0)
        cp.report()
        return cp


# =============================================================
# 5. PROBLÈME D'ÉCOULEMENT
# =============================================================

def test_flow():
    """Test du problème d'écoulement avec données CSV."""
    print("\n" + "=" * 70)
    print("  PROBLÈME D'ÉCOULEMENT")
    print("=" * 70)
    
    csv_path = "data/flow.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        fp = FlowProblem(df["x"].values, df["v"].values)
        fp.report()
        return fp
    else:
        print(f"\n  ⚠️  Fichier {csv_path} introuvable.")
        print("  Utilisation de données synthétiques...\n")
        
        # Données synthétiques
        x_data = np.array([0, 1, 2, 3, 4, 5, 6])
        v_data = np.array([0.5, 0.8, 1.1, 1.3, 1.4, 1.3, 1.0])
        fp = FlowProblem(x_data, v_data)
        fp.report()
        return fp


# =============================================================
# 6. VISUALISATIONS
# =============================================================

def generate_visualizations(cp, fp, runge_data=None, errors_data=None):
    """Génère tous les graphes de visualisation."""
    print("\n" + "=" * 70)
    print("  GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)
    
    os.makedirs("results", exist_ok=True)
    viz = Visualizer(figsize=(12, 6))
    
    # ─── Refroidissement ──────────────────────────────────────
    print("\n  📊 Cooling Problem...")
    
    # Points fins pour tracé lisse
    t_fine = np.linspace(cp.t_data[0], cp.t_data[-1], 300)
    T_interp = cp.temperature(t_fine, method='newton')
    
    # Modèle exponentiel optimal
    k_opt = cp.estimate_k()
    T_model = cp.exponential_model(t_fine, k_opt)
    
    # Graphe complet
    fig1 = viz.plot_cooling_analysis(cp.t_data, cp.T_data, t_fine, T_interp, k_opt, T_model)
    fig1.savefig("results/01_cooling_analysis.png", dpi=300, bbox_inches='tight')
    print("     ✅ 01_cooling_analysis.png")
    
    # Comparaison interpolations
    T_lag = cp.temperature(t_fine, method='lagrange')
    fig2 = viz.plot_interpolation_comparison(
        cp.t_data, cp.T_data,
        {"Lagrange": T_lag, "Newton": T_interp},
        t_fine,
        title="Interpolation — Refroidissement"
    )
    fig2.savefig("results/02_cooling_interpolations.png", dpi=300, bbox_inches='tight')
    print("     ✅ 02_cooling_interpolations.png")
    
    # ─── Écoulement ───────────────────────────────────────────
    print("\n  📊 Flow Problem...")
    
    # Points fins pour tracé
    x_fine = np.linspace(fp.x_data[0], fp.x_data[-1], 300)
    v_interp = fp.velocity(x_fine, method='newton')
    
    # Graphe complet
    fig3 = viz.plot_flow_analysis(fp.x_data, fp.v_data, x_fine, v_interp, fp.width_func)
    fig3.savefig("results/03_flow_analysis.png", dpi=300, bbox_inches='tight')
    print("     ✅ 03_flow_analysis.png")
    
    # Comparaison interpolations
    v_lag = fp.velocity(x_fine, method='lagrange')
    fig4 = viz.plot_interpolation_comparison(
        fp.x_data, fp.v_data,
        {"Lagrange": v_lag, "Newton": v_interp},
        x_fine,
        title="Interpolation — Écoulement"
    )
    fig4.savefig("results/04_flow_interpolations.png", dpi=300, bbox_inches='tight')
    print("     ✅ 04_flow_interpolations.png")
    
    # ─── Convergence intégration (Cooling) ────────────────────
    print("\n  📊 Convergence (Cooling)...")
    
    Q_ref = cp.total_heat_loss(method='adaptive')
    n_values = [5, 10, 20, 50, 100, 200, 500]
    errors = {
        "Trapèze": [abs(cp.total_heat_loss(method='trapeze', n=n) - Q_ref) for n in n_values],
        "Simpson": [abs(cp.total_heat_loss(method='simpson', n=n) - Q_ref) for n in n_values],
    }
    
    fig5 = viz.plot_convergence(
        n_values, errors,
        title="Convergence — Chaleur dissipée",
        ylabel="Erreur absolue |Q_n - Q_ref|"
    )
    fig5.savefig("results/05_cooling_convergence.png", dpi=300, bbox_inches='tight')
    print("     ✅ 05_cooling_convergence.png")
    
    # ─── Convergence intégration (Flow) ───────────────────────
    print("\n  📊 Convergence (Flow)...")
    
    D_ref = fp.total_flow_rate(method='adaptive')
    errors = {
        "Trapèze": [abs(fp.total_flow_rate(method='trapeze', n=n) - D_ref) for n in n_values],
        "Simpson": [abs(fp.total_flow_rate(method='simpson', n=n) - D_ref) for n in n_values],
    }
    
    fig6 = viz.plot_convergence(
        n_values, errors,
        title="Convergence — Débit volumique",
        ylabel="Erreur absolue |D_n - D_ref|"
    )
    fig6.savefig("results/06_flow_convergence.png", dpi=300, bbox_inches='tight')
    print("     ✅ 06_flow_convergence.png")
    
    # ─── Runge  ─────────────────────────────
    if runge_data:
        print("\n  📊 Runge (Travail des amis)...")
        fig7 = viz.plot_runge_phenomenon(runge_data["x_fine"], runge_data["y_true"], runge_data)
        fig7.savefig("results/07_runge_phenomenon.png", dpi=300, bbox_inches='tight')
        print("     ✅ 07_runge_phenomenon.png")
    
    # ─── Convergence générale  ──────────────
    if errors_data:
        print("\n  📊 Convergence générale (Travail des amis)...")
        errors_dict = {
            "Rectangle":   errors_data["rectangle"],
            "Trapèze":     errors_data["trapezoidal"],
            "Simpson":     errors_data["simpson"],
        }
        fig8 = viz.plot_convergence(
            errors_data["n"],
            errors_dict,
            title="Convergence — ∫₀¹ eˣ dx (tous les amis)"
        )
        fig8.savefig("results/08_convergence_general.png", dpi=300, bbox_inches='tight')
        print("     ✅ 08_convergence_general.png")
    
    # ─── Comparaison des méthodes d'intégration (Travail des amis) ─────
    print("\n  📊 Comparaison des méthodes (Travail des amis)...")
    fig9 = viz.plot_integration_comparison(
        f=math.exp,
        a=0, b=1,
        n_values=[2, 4, 8, 16, 32, 64, 128],
        methods_dict={
            "Rectangle":  NewtonCotes.rectangle,
            "Trapèze":    NewtonCotes.trapezoidal,
            "Simpson":    NewtonCotes.simpson,
        },
        exact_value=math.e - 1,
        title="Comparaison des méthodes — ∫₀¹ eˣ dx"
    )
    fig9.savefig("results/09_methods_comparison.png", dpi=300, bbox_inches='tight')
    print("     ✅ 09_methods_comparison.png")
    
    print("\n  ✅ Toutes les visualisations générées dans 'results/'")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PROJET ANALYSE NUMÉRIQUE")
    print("  Interpolation Polynomiale + Intégration + Problèmes Réels")
    print("=" * 70)
    
    # Tests de base
    test_simple()
    
   
    runge_data = test_runge()
    errors_data = test_convergence()
    
    # Problèmes réels
    cp = test_cooling()
    fp = test_flow()
    
    # Visualisations (incluant les graphes des amis)
    generate_visualizations(cp, fp, runge_data, errors_data)
    
    print("\n" + "=" * 70)
    print("  ✅ PROGRAMME TERMINÉ AVEC SUCCÈS")
    print("=" * 70 + "\n")