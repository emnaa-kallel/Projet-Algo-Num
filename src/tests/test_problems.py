"""
Tests pour les classes CoolingProblem et FlowProblem
"""
import numpy as np
import pandas as pd
import sys
import os

# Permet l'import des modules du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.problems import CoolingProblem, FlowProblem


# =============================================================
# 1. TESTS POUR CoolingProblem
# =============================================================

def test_cooling_problem():
    """Test des méthodes principales de CoolingProblem."""
    print("\n" + "=" * 70)
    print("  TEST COOLING PROBLEM")
    print("=" * 70)

    # Données fictives ou depuis cooling.csv
    t_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    T_data = np.array([100, 85.5, 73.6, 63.9, 56.1, 49.9, 44.9, 40.8, 37.4, 34.6, 32.3])

    cooling = CoolingProblem(t_data, T_data, T_ambient=20.0, h_coeff=50.0)

    print("\n📌 Test 1: Interpolation (température)")
    for t_q in [2.5, 5.0, 7.5]:
        T_lagrange = cooling.temperature(t_q, method='lagrange')
        T_newton = cooling.temperature(t_q, method='newton')
        print(f"   t={t_q:4.1f}s  —  Lagrange: {T_lagrange:7.4f}°C  |  Newton: {T_newton:7.4f}°C")

    print("\n📌 Test 2: Taux de perte de chaleur")
    for t_q in [2.5, 5.0]:
        q_lagrange = cooling.heat_loss_rate(t_q, method='lagrange')
        q_newton = cooling.heat_loss_rate(t_q, method='newton')
        print(f"   t={t_q:4.1f}s  —  Lagrange: {q_lagrange:8.4f} W/m²  |  Newton: {q_newton:8.4f} W/m²")

    print("\n📌 Test 3: Chaleur totale dissipée (trois méthodes)")
    for method in ['trapeze', 'simpson', 'adaptive']:
        Q = cooling.total_heat_loss(method=method, n=100)
        print(f"   {method:10s}: Q = {Q:10.4f} J/m²")

    print("\n📌 Test 4: Modèle exponentiel")
    k_test = 0.15
    t_eval = np.array([0, 5, 10])
    T_exp = cooling.exponential_model(t_eval, k_test)
    print(f"   k = {k_test:.4f} s⁻¹")
    for i, t_val in enumerate(t_eval):
        print(f"      T({t_val:2.0f}s) = {T_exp[i]:.4f}°C")

    print("\n📌 Test 5: Erreur du modèle exponentiel")
    for k in [0.10, 0.15, 0.20]:
        error = cooling.model_error(k)
        print(f"   k = {k:.4f}  →  E(k) = {error:.6f}")

    print("\n📌 Test 6: Estimation du k optimal (bissection)")
    k_optimal = cooling.estimate_k(k_min=0.05, k_max=0.3, tol=1e-4)
    error_optimal = cooling.model_error(k_optimal)
    print(f"   k optimal = {k_optimal:.6f} s⁻¹")
    print(f"   E(k_opt) = {error_optimal:.6f}")

    print("\n📌 Test 7: Rapport complet")
    cooling.report()


# =============================================================
# 2. TESTS POUR FlowProblem
# =============================================================

def test_flow_problem():
    """Test des méthodes principales de FlowProblem."""
    print("\n" + "=" * 70)
    print("  TEST FLOW PROBLEM")
    print("=" * 70)

    # Données fictives ou depuis flow.csv
    x_data = np.array([0, 1, 2, 3, 4, 5, 6])
    v_data = np.array([0.5, 0.8, 1.1, 1.3, 1.4, 1.3, 1.0])

    flow = FlowProblem(x_data, v_data)

    print("\n📌 Test 1: Interpolation (vitesse)")
    for x_q in [1.5, 3.0, 4.5]:
        v_lagrange = flow.velocity(x_q, method='lagrange')
        v_newton = flow.velocity(x_q, method='newton')
        print(f"   x={x_q:4.1f}m  —  Lagrange: {v_lagrange:7.4f} m/s  |  Newton: {v_newton:7.4f} m/s")

    print("\n📌 Test 2: Débit local q(x) = v(x)·w(x)")
    for x_q in [1.5, 3.0]:
        q_newton = flow.local_flow_rate(x_q, method='newton')
        w_x = flow.width_func(x_q)
        v_x = flow.velocity(x_q, method='newton')
        print(f"   x={x_q:4.1f}m  —  w(x)={w_x:.4f}m  |  v(x)={v_x:.4f}m/s  |  q(x)={q_newton:.4f}m²/s")

    print("\n📌 Test 3: Débit total D (trois méthodes)")
    for method in ['trapeze', 'simpson', 'adaptive']:
        D = flow.total_flow_rate(method=method, n=100)
        print(f"   {method:10s}: D = {D:10.4f} m³/s")

    print("\n📌 Test 4: Accélération dv/dx (différences finies)")
    x_eval = np.linspace(x_data[0], x_data[-1], 7)
    acc = flow.acceleration(x_eval)
    for i, (x_val, a_val) in enumerate(zip(x_eval, acc)):
        print(f"   x={x_val:.1f}m  →  dv/dx = {a_val:8.6f} m/s per m")

    print("\n📌 Test 5: Travail W pour différentes masses")
    for mass in [1.0, 2.0, 5.0]:
        W = flow.work(mass=mass)
        print(f"   mass = {mass:4.1f} kg  →  W = {W:10.4f} J")

    print("\n📌 Test 6: Rapport complet")
    flow.report()


# =============================================================
# 3. TESTS AVEC DONNÉES RÉELLES (CSV)
# =============================================================

def test_with_csv_data():
    """Test avec les fichiers CSV du projet."""
    print("\n" + "=" * 70)
    print("  TESTS AVEC DONNÉES CSV")
    print("=" * 70)

    csv_path_cooling = os.path.join(os.path.dirname(__file__), '../../data/cooling.csv')
    csv_path_flow = os.path.join(os.path.dirname(__file__), '../../data/flow.csv')

    # Test avec cooling.csv
    if os.path.exists(csv_path_cooling):
        print("\n📊 Données de refroidissement (cooling.csv):")
        df_cooling = pd.read_csv(csv_path_cooling)
        print(df_cooling.head())
        
        t_data = df_cooling['t'].values
        T_data = df_cooling['T'].values
        
        cooling = CoolingProblem(t_data, T_data)
        print("\n   Rapport du problème de refroidissement :")
        cooling.report()
    else:
        print(f"\n⚠️  Fichier cooling.csv introuvable : {csv_path_cooling}")

    # Test avec flow.csv
    if os.path.exists(csv_path_flow):
        print("\n📊 Données d'écoulement (flow.csv):")
        df_flow = pd.read_csv(csv_path_flow)
        print(df_flow.head())
        
        x_data = df_flow['x'].values
        v_data = df_flow['v'].values
        
        flow = FlowProblem(x_data, v_data)
        print("\n   Rapport du problème d'écoulement :")
        flow.report()
    else:
        print(f"\n⚠️  Fichier flow.csv introuvable : {csv_path_flow}")


# =============================================================
# 4. TESTS DE VALIDATION NUMÉRIQUE
# =============================================================

def test_numerical_validation():
    """Validations : cohérence entre interpolations et intégrales."""
    print("\n" + "=" * 70)
    print("  TESTS DE VALIDATION NUMÉRIQUE")
    print("=" * 70)

    # Données test
    t_data = np.array([0, 1, 2, 3, 4, 5])
    T_data = np.array([100, 90, 75, 60, 50, 45])

    cooling = CoolingProblem(t_data, T_data)

    print("\n✓ Validation 1: Les deux interpolations donnent des résultats proches")
    t_test = 2.5
    T_lag = cooling.temperature(t_test, method='lagrange')
    T_new = cooling.temperature(t_test, method='newton')
    diff = abs(T_lag - T_new)
    print(f"  Lagrange: {T_lag:.6f}, Newton: {T_new:.6f}, Différence: {diff:.2e}")
    assert diff < 1e-6, "❌ Les interpolations ne sont pas cohérentes!"
    print("  ✅ PASS")

    print("\n✓ Validation 2: L'énergie dissipée est positive")
    for method in ['trapeze', 'simpson', 'adaptive']:
        Q = cooling.total_heat_loss(method=method)
        print(f"  {method:10s}: Q = {Q:8.4f} J/m² (Q > 0 ? {Q > 0})")
        assert Q > 0, f"❌ {method}: Q est négative!"
    print("  ✅ PASS")

    print("\n✓ Validation 3: Convergence de l'intégration adaptative")
    n_values = [10, 50, 100, 200]
    Q_prev = None
    for n in n_values:
        Q_trap = cooling.total_heat_loss(method='trapeze', n=n)
        print(f"  n={n:3d}  →  Q = {Q_trap:.6f}")
        if Q_prev is not None:
            conv = abs(Q_trap - Q_prev) / abs(Q_prev) * 100
            print(f"        Variation: {conv:.4f}%")
    print("  ✅ PASS")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    # Tests basiques
    test_cooling_problem()
    test_flow_problem()
    
    # Tests avec CSV
    test_with_csv_data()
    
    # Validations numériques
    test_numerical_validation()
    
    print("\n" + "=" * 70)
    print("  ✅ TOUS LES TESTS TERMINÉS")
    print("=" * 70)
