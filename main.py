"""
=============================================================
  Projet Analyse Numérique — main.py
  Interpolation + Intégration + Problèmes réels
=============================================================

Structure des fichiers :
  src/visualization.py  ← Visualizer (fusion de visualizer.py
                                       et visualize_problems.py)
  main.py               ← point d'entrée unique
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
from src.visualization import Visualizer   # ← source unique


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
        lag = interp.evaluate(x_q, "lagrange")
        new = interp.evaluate(x_q, "newton")
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
    """Test du phénomène de Runge — nœuds équidistants vs Chebyshev."""
    print("\n" + "=" * 70)
    print("  PHÉNOMÈNE DE RUNGE — équidistants vs Tchebychev")
    print("=" * 70)

    try:
        def runge_func(x):
            return 1 / (1 + 25 * x**2)

        n = 10
        x_eq   = np.linspace(-1, 1, n + 1)
        x_cheb = np.cos(np.pi * (2 * np.arange(n + 1) + 1) / (2 * (n + 1)))

        y_eq   = runge_func(x_eq)
        y_cheb = runge_func(x_cheb)

        interp_eq   = PolynomialInterpolation(x_eq, y_eq)
        interp_cheb = PolynomialInterpolation(x_cheb, y_cheb)

        x_fine = np.linspace(-1, 1, 500)
        y_true        = runge_func(x_fine)
        y_eq_interp   = interp_eq.evaluate(x_fine,   method="newton")
        y_cheb_interp = interp_cheb.evaluate(x_fine, method="newton")

        print("  📌 Interpolation avec n=10 points")
        print(f"     Équidistants : x ∈ [{x_eq[0]:.1f}, {x_eq[-1]:.1f}]")
        print(f"     Chebyshev    : x ∈ [{x_cheb[0]:.1f}, {x_cheb[-1]:.1f}]")

        err_eq   = np.max(np.abs(y_true - y_eq_interp))
        err_cheb = np.max(np.abs(y_true - y_cheb_interp))
        print(f"     Erreur max équidistant : {err_eq:.2e}")
        print(f"     Erreur max Chebyshev   : {err_cheb:.2e}")

        return {
            "x_fine":      x_fine,
            "y_true":      y_true,
            "Équidistant": y_eq_interp,
            "Chebyshev":   y_cheb_interp,
        }

    except Exception as e:
        print(f"  ⚠️  Erreur dans test_runge : {e}")
        return None


# =============================================================
# 3. ANALYSE DE CONVERGENCE — ∫₀¹ eˣ dx
# =============================================================

def test_convergence():
    """Analyse de convergence pour ∫₀¹ eˣ dx."""
    print("\n" + "=" * 70)
    print("  CONVERGENCE — ∫₀¹ eˣ dx (exact = e-1)")
    print("=" * 70)

    try:
        def f(x):
            return np.exp(x)

        a, b  = 0, 1
        exact = math.e - 1
        n_values = [2, 4, 8, 16, 32, 64, 128, 256]

        errors = {"n": n_values}
        for method_name, method_func in [
            ("rectangle",   NewtonCotes.rectangle),
            ("trapezoidal", NewtonCotes.trapezoidal),
            ("simpson",     NewtonCotes.simpson),
        ]:
            errs = [abs(method_func(f, a, b, n) - exact) for n in n_values]
            errors[method_name] = errs
            print(f"  📌 {method_name.capitalize():12s}: erreur pour n=256 = {errs[-1]:.2e}")

        return errors

    except Exception as e:
        print(f"  ⚠️  Erreur dans test_convergence : {e}")
        return None


# =============================================================
# 4. PROBLÈME DE REFROIDISSEMENT
# =============================================================

def test_cooling():
    """Test du problème de refroidissement (CSV ou données synthétiques)."""
    print("\n" + "=" * 70)
    print("  PROBLÈME DE REFROIDISSEMENT")
    print("=" * 70)

    csv_path = "data/cooling.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        cp = CoolingProblem(df["t"].values, df["T"].values,
                            T_ambient=20.0, h_coeff=50.0)
    else:
        print(f"\n  ⚠️  Fichier {csv_path} introuvable — données synthétiques.\n")
        t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        T_data = np.array([100, 85.5, 73.6, 63.9, 56.1, 49.9,
                           44.9, 40.8, 37.4, 34.6, 32.3])
        cp = CoolingProblem(t_data, T_data, T_ambient=20.0, h_coeff=50.0)

    cp.report()
    return cp


# =============================================================
# 5. PROBLÈME D'ÉCOULEMENT
# =============================================================

def test_flow():
    """Test du problème d'écoulement (CSV ou données synthétiques)."""
    print("\n" + "=" * 70)
    print("  PROBLÈME D'ÉCOULEMENT")
    print("=" * 70)

    csv_path = "data/flow.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        fp = FlowProblem(df["x"].values, df["v"].values)
    else:
        print(f"\n  ⚠️  Fichier {csv_path} introuvable — données synthétiques.\n")
        x_data = np.array([0, 1, 2, 3, 4, 5, 6])
        v_data = np.array([0.5, 0.8, 1.1, 1.3, 1.4, 1.3, 1.0])
        fp = FlowProblem(x_data, v_data)

    fp.report()
    return fp


# =============================================================
# 6. VISUALISATIONS (ex-visualize_problems.py intégré ici)
# =============================================================

def generate_visualizations(cp, fp, runge_data=None, errors_data=None):
    """
    Génère et sauvegarde tous les graphes dans results/.

    Intègre les fonctions de l'ancien visualize_problems.py :
      - visualize_cooling_problem          → 01_cooling_analysis.png
      - visualize_cooling_interpolations   → 02_cooling_interpolations.png
      - visualize_flow_problem             → 03_flow_analysis.png
      - visualize_flow_interpolations      → 04_flow_interpolations.png
      - visualize_cooling_heat_convergence → 05_cooling_convergence.png
      - visualize_flow_debit_convergence   → 06_flow_convergence.png
      - plot_runge_phenomenon              → 07_runge_phenomenon.png  (si dispo)
      - plot_convergence général           → 08_convergence_general.png (si dispo)
      - plot_integration_comparison        → 09_methods_comparison.png
    """
    print("\n" + "=" * 70)
    print("  GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    n_values = [5, 10, 20, 50, 100, 200, 500]

    # ── Refroidissement ───────────────────────────────────────────────────────
    print("\n  📊 Cooling — analyse principale...")
    viz = Visualizer(figsize=(12, 6))

    t_fine   = np.linspace(cp.t_data[0], cp.t_data[-1], 300)
    T_newton = cp.temperature(t_fine, method="newton")
    k_opt    = cp.estimate_k()
    T_model  = cp.exponential_model(t_fine, k_opt)

    fig1 = viz.plot_cooling_analysis(cp.t_data, cp.T_data, t_fine, T_newton, k_opt, T_model)
    fig1.savefig("results/01_cooling_analysis.png", dpi=300, bbox_inches="tight")
    print("     ✅ 01_cooling_analysis.png")

    # ── Comparaison interpolations — Cooling ──────────────────────────────────
    print("  📊 Cooling — comparaison interpolations...")
    T_lag = cp.temperature(t_fine, method="lagrange")
    fig2 = viz.plot_interpolation_comparison(
        cp.t_data, cp.T_data,
        {"Lagrange": T_lag, "Newton": T_newton},
        t_fine,
        title="Comparaison des interpolations — Refroidissement",
    )
    fig2.savefig("results/02_cooling_interpolations.png", dpi=300, bbox_inches="tight")
    print("     ✅ 02_cooling_interpolations.png")

    # ── Écoulement — analyse principale ──────────────────────────────────────
    print("\n  📊 Flow — analyse principale...")
    viz_flow = Visualizer(figsize=(14, 5))

    x_fine   = np.linspace(fp.x_data[0], fp.x_data[-1], 300)
    v_newton = fp.velocity(x_fine, method="newton")

    fig3 = viz_flow.plot_flow_analysis(fp.x_data, fp.v_data, x_fine, v_newton, fp.width_func)
    fig3.savefig("results/03_flow_analysis.png", dpi=300, bbox_inches="tight")
    print("     ✅ 03_flow_analysis.png")

    # ── Comparaison interpolations — Flow ────────────────────────────────────
    print("  📊 Flow — comparaison interpolations...")
    v_lag = fp.velocity(x_fine, method="lagrange")
    fig4 = viz.plot_interpolation_comparison(
        fp.x_data, fp.v_data,
        {"Lagrange": v_lag, "Newton": v_newton},
        x_fine,
        title="Comparaison des interpolations — Écoulement",
    )
    fig4.savefig("results/04_flow_interpolations.png", dpi=300, bbox_inches="tight")
    print("     ✅ 04_flow_interpolations.png")

    # ── Convergence intégration — Cooling ────────────────────────────────────
    print("\n  📊 Convergence (Cooling — chaleur dissipée)...")
    Q_ref      = cp.total_heat_loss(method="adaptive")
    errors_cool = {
        "Trapèze": [abs(cp.total_heat_loss(method="trapeze",  n=n) - Q_ref) for n in n_values],
        "Simpson":  [abs(cp.total_heat_loss(method="simpson", n=n) - Q_ref) for n in n_values],
    }
    fig5 = viz.plot_convergence(
        n_values, errors_cool,
        title="Convergence de l'intégration — Chaleur dissipée",
        xlabel="n (subdivisions)",
        ylabel="Erreur absolue |Q_n - Q_ref|",
    )
    fig5.savefig("results/05_cooling_convergence.png", dpi=300, bbox_inches="tight")
    print("     ✅ 05_cooling_convergence.png")

    # ── Convergence intégration — Flow ───────────────────────────────────────
    print("  📊 Convergence (Flow — débit volumique)...")
    D_ref      = fp.total_flow_rate(method="adaptive")
    errors_flow = {
        "Trapèze": [abs(fp.total_flow_rate(method="trapeze",  n=n) - D_ref) for n in n_values],
        "Simpson":  [abs(fp.total_flow_rate(method="simpson", n=n) - D_ref) for n in n_values],
    }
    fig6 = viz.plot_convergence(
        n_values, errors_flow,
        title="Convergence de l'intégration — Débit volumique",
        xlabel="n (subdivisions)",
        ylabel="Erreur absolue |D_n - D_ref|",
    )
    fig6.savefig("results/06_flow_convergence.png", dpi=300, bbox_inches="tight")
    print("     ✅ 06_flow_convergence.png")

    # ── Phénomène de Runge ───────────────────────────────────────────────────
    if runge_data:
        print("\n  📊 Phénomène de Runge...")
        # plot_runge_phenomenon sauvegarde lui-même dans results/
        viz.plot_runge_phenomenon(
            runge_data["x_fine"],
            runge_data["y_true"],
            runge_data,
        )
        print("     ✅ 07_runge_phenomenon.png")

    # ── Convergence générale ─────────────────────────────────────────────────
    if errors_data:
        print("\n  📊 Convergence générale...")
        errors_dict = {
            "Rectangle": errors_data["rectangle"],
            "Trapèze":   errors_data["trapezoidal"],
            "Simpson":   errors_data["simpson"],
        }
        fig8 = viz.plot_convergence(
            errors_data["n"],
            errors_dict,
            title="Convergence — ∫[0,1] eˣ dx",
        )
        fig8.savefig("results/08_convergence_general.png", dpi=300, bbox_inches="tight")
        print("     ✅ 08_convergence_general.png")

    # ── Comparaison des méthodes d'intégration ───────────────────────────────
    print("\n  📊 Comparaison des méthodes d'intégration...")
    fig9 = viz.plot_integration_comparison(
        f=math.exp,
        a=0,
        b=1,
        n_values=[2, 4, 8, 16, 32, 64, 128],
        methods_dict={
            "Rectangle": NewtonCotes.rectangle,
            "Trapèze":   NewtonCotes.trapezoidal,
            "Simpson":   NewtonCotes.simpson,
        },
        exact_value=math.e - 1,
        title="Comparaison des méthodes — ∫[0,1] eˣ dx",
    )
    fig9.savefig("results/09_methods_comparison.png", dpi=300, bbox_inches="tight")
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

    # Tests numériques
    test_simple()
    runge_data  = test_runge()
    errors_data = test_convergence()

    # Problèmes réels
    cp = test_cooling()
    fp = test_flow()

    # Toutes les visualisations (cooling + flow + runge + convergence)
    generate_visualizations(cp, fp, runge_data, errors_data)

    print("\n" + "=" * 70)
    print("  ✅ PROGRAMME TERMINÉ AVEC SUCCÈS")
    print("=" * 70 + "\n")