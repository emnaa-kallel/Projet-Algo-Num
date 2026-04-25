"""
Programme principal : étude de la convergence des méthodes d'intégration.

Ce script compare la précision des méthodes de Newton-Cotes, de l'intégration
adaptative et de la quadrature de Gauss-Legendre pour plusieurs fonctions tests,
en faisant varier le nombre de sous-intervalles et en analysant la convergence.

Fonctions tests utilisées :
    f1(x) = sin(x)        sur [0, π]    → valeur exacte : 2
    f2(x) = exp(x)        sur [0, 1]    → valeur exacte : e - 1
    f3(x) = 1/(1 + x²)   sur [0, 1]    → valeur exacte : π/4
    f4(x) = x * ln(x+1)  sur [0, 2]    → valeur exacte : 2*ln(3) - 3/2
"""

import math
import os

import matplotlib.pyplot as plt

from src.integration import AdaptiveIntegration, GaussQuadrature, NewtonCotes

# ─────────────────────────────────────────────────────────────────
# Définition des fonctions tests et de leurs valeurs exactes
# ─────────────────────────────────────────────────────────────────

TEST_FUNCTIONS = [
    {
        "name": "exp(x) sur [0, 1]",
        "f": math.exp,
        "a": 0.0,
        "b": 1.0,
        "exact": math.e - 1,
    },
    # Fonction test pour l'annexe A.3 : f(x) = e^x sur [0,1]
    # Valeur exacte: ∫₀¹ e^x dx = e - 1 ≈ 1.718281828459045
]


# Valeurs de n testées pour les méthodes Newton-Cotes
N_VALUES = [2, 4, 8, 16, 32, 64, 128]


# ─────────────────────────────────────────────────────────────────
# Fonctions utilitaires d'affichage
# ─────────────────────────────────────────────────────────────────

def print_separator(char="-", width=72):
    """Affiche une ligne de séparation."""
    print(char * width)


def print_header(title):
    """Affiche un en-tête de section."""
    print_separator("=")
    print(f"  {title}")
    print_separator("=")


def print_subheader(title):
    """Affiche un sous-titre de section."""
    print_separator()
    print(f"  {title}")
    print_separator()


def format_error(error):
    """Formate une erreur absolue en notation scientifique."""
    return f"{error:.2e}"


# ─────────────────────────────────────────────────────────────────
# Étude de convergence — méthodes Newton-Cotes
# ─────────────────────────────────────────────────────────────────

def study_newton_cotes(func_info):
    """
    Étudie la convergence des méthodes de Newton-Cotes pour une fonction donnée.

    Affiche un tableau comparatif des erreurs absolues en fonction de n
    pour les méthodes rectangle, trapèzes et Simpson 1/3.

    Paramètres
    ----------
    func_info : dict
        Dictionnaire avec les clés 'name', 'f', 'a', 'b', 'exact'.
    """
    f = func_info["f"]
    a = func_info["a"]
    b = func_info["b"]
    exact = func_info["exact"]

    print_subheader(f"Fonction : {func_info['name']}  (valeur exacte = {exact:.10f})")

    col_w = 14
    header = (
        f"{'n':>6}  "
        f"{'Rectangle':>{col_w}}  "
        f"{'Trapèzes':>{col_w}}  "
        f"{'Simpson 1/3':>{col_w}}  "
        f"{'Simpson 3/8':>{col_w}}"
    )
    print(header)
    print_separator("-")

    for n in N_VALUES:
        # Simpson 3/8 nécessite n multiple de 3 → ajustement
        n38 = n if n % 3 == 0 else n + (3 - n % 3)

        rect = NewtonCotes.rectangle(f, a, b, n)
        trap = NewtonCotes.trapezoidal(f, a, b, n)
        simp = NewtonCotes.simpson(f, a, b, n)
        simp38 = NewtonCotes.simpson_38(f, a, b, n38)

        err_rect = abs(rect - exact)
        err_trap = abs(trap - exact)
        err_simp = abs(simp - exact)
        err_s38 = abs(simp38 - exact)

        print(
            f"{n:>6}  "
            f"{format_error(err_rect):>{col_w}}  "
            f"{format_error(err_trap):>{col_w}}  "
            f"{format_error(err_simp):>{col_w}}  "
            f"{format_error(err_s38):>{col_w}}"
        )


# ─────────────────────────────────────────────────────────────────
# Étude de convergence — intégration adaptative
# ─────────────────────────────────────────────────────────────────

def study_adaptive(func_info):
    """
    Étudie la précision de la méthode de Simpson adaptative pour
    différentes tolérances.

    Paramètres
    ----------
    func_info : dict
        Dictionnaire avec les clés 'name', 'f', 'a', 'b', 'exact'.
    """
    f = func_info["f"]
    a = func_info["a"]
    b = func_info["b"]
    exact = func_info["exact"]

    tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

    print_subheader(f"Fonction : {func_info['name']}  (valeur exacte = {exact:.10f})")
    print(f"{'Tolérance':>12}  {'Résultat':>18}  {'Erreur absolue':>16}")
    print_separator("-")

    for tol in tolerances:
        integrator = AdaptiveIntegration(tol=tol)
        result = integrator.adaptive_simpson(f, a, b)
        error = abs(result - exact)
        print(f"{tol:>12.2e}  {result:>18.12f}  {format_error(error):>16}")


# ─────────────────────────────────────────────────────────────────
# Comparaison avec la quadrature de Gauss
# ─────────────────────────────────────────────────────────────────

def study_gauss(func_info):
    """
    Compare la quadrature de Gauss-Legendre (2 et 3 points) avec
    les méthodes Newton-Cotes de référence pour une fonction donnée.

    Paramètres
    ----------
    func_info : dict
        Dictionnaire avec les clés 'name', 'f', 'a', 'b', 'exact'.
    """
    f = func_info["f"]
    a = func_info["a"]
    b = func_info["b"]
    exact = func_info["exact"]

    print_subheader(f"Fonction : {func_info['name']}  (valeur exacte = {exact:.10f})")

    methods = [
        ("Rectangle  (n=1)", NewtonCotes.rectangle(f, a, b, 1)),
        ("Trapèzes   (n=1)", NewtonCotes.trapezoidal(f, a, b, 1)),
        ("Simpson    (n=2)", NewtonCotes.simpson(f, a, b, 2)),
        ("Gauss-2pts      ", GaussQuadrature.gauss_legendre_2(f, a, b)),
        ("Gauss-3pts      ", GaussQuadrature.gauss_legendre_3(f, a, b)),
    ]

    print(f"{'Méthode':>20}  {'Résultat':>18}  {'Erreur':>14}")
    print_separator("-")
    for name, result in methods:
        error = abs(result - exact)
        print(f"{name:>20}  {result:>18.12f}  {format_error(error):>14}")


# ─────────────────────────────────────────────────────────────────
# Analyse des ordres de convergence
# ─────────────────────────────────────────────────────────────────

def estimate_convergence_order(func_info):
    """
    Estime empiriquement l'ordre de convergence de chaque méthode
    en calculant le rapport log(err_n / err_2n) / log(2).

    Un ordre ≈ 2 pour les trapèzes, ≈ 4 pour Simpson est attendu.

    Paramètres
    ----------
    func_info : dict
        Dictionnaire avec les clés 'name', 'f', 'a', 'b', 'exact'.
    """
    f = func_info["f"]
    a = func_info["a"]
    b = func_info["b"]
    exact = func_info["exact"]

    print_subheader(f"Ordres de convergence — {func_info['name']}")

    ns = [4, 8, 16, 32, 64]

    def order(err1, err2):
        """Calcule l'ordre empirique entre deux niveaux successifs."""
        if err1 == 0 or err2 == 0:
            return float("inf")
        return math.log(err1 / err2) / math.log(2)

    print(f"{'n':>6}  {'p_trap':>10}  {'p_simp':>10}")
    print_separator("-")

    prev_trap = abs(NewtonCotes.trapezoidal(f, a, b, ns[0]) - exact)
    prev_simp = abs(NewtonCotes.simpson(f, a, b, ns[0]) - exact)

    for n in ns[1:]:
        curr_trap = abs(NewtonCotes.trapezoidal(f, a, b, n) - exact)
        curr_simp = abs(NewtonCotes.simpson(f, a, b, n) - exact)

        p_trap = order(prev_trap, curr_trap)
        p_simp = order(prev_simp, curr_simp)

        print(f"{n:>6}  {p_trap:>10.3f}  {p_simp:>10.3f}")

        prev_trap = curr_trap
        prev_simp = curr_simp


# ─────────────────────────────────────────────────────────────────
# Analyse complète (section 4.2 du projet)
# ─────────────────────────────────────────────────────────────────

def analyze_integration_convergence():
    """
    Analyse complète de l'intégration numérique (section 4.2 du projet).
    
    Cette fonction réalise :
    1. Calcul des erreurs pour f(x)=e^x sur [0,1]
    2. Tracé des courbes d'erreur en échelle log-log
    3. Comparaison adaptive vs Simpson composé (n=100)
    """
    import os
    from src.visualization import Visualizer
    
    print_subheader("ANALYSE COMPLÈTE - Section 4.2")
    
    # Fonction test : f(x) = e^x sur [0, 1]
    f = math.exp
    a, b = 0.0, 1.0
    exact = math.e - 1  # ≈ 1.718281828459045
    
    print(f"Fonction test: f(x) = e^x sur [0, 1]")
    print(f"Valeur exacte: {exact:.15f}")
    print()
    
    # Valeurs de n testées
    n_values = [2, 4, 8, 16, 32, 64, 128, 256]
    
    # Calcul des erreurs pour chaque méthode
    errors_rect = []
    errors_trap = []
    errors_simp = []
    errors_s38 = []
    
    print(f"{'n':>6}  {'Rectangle':>14}  {'Trapèzes':>14}  {'Simpson 1/3':>14}  {'Simpson 3/8':>14}")
    print_separator("-")
    
    for n in n_values:
        # Rectangle (point milieu)
        rect = NewtonCotes.rectangle(f, a, b, n)
        # Trapèzes
        trap = NewtonCotes.trapezoidal(f, a, b, n)
        # Simpson 1/3
        simp = NewtonCotes.simpson(f, a, b, n)
        # Simpson 3/8
        n38 = n if n % 3 == 0 else n + (3 - n % 3)
        simp38 = NewtonCotes.simpson_38(f, a, b, n38)
        
        err_rect = abs(rect - exact)
        err_trap = abs(trap - exact)
        err_simp = abs(simp - exact)
        err_s38 = abs(simp38 - exact)
        
        errors_rect.append(err_rect)
        errors_trap.append(err_trap)
        errors_simp.append(err_simp)
        errors_s38.append(err_s38)
        
        print(f"{n:>6}  {err_rect:>14.2e}  {err_trap:>14.2e}  {err_simp:>14.2e}  {err_s38:>14.2e}")
    
    print()
    
    # Tracé des courbes de convergence
    visualizer = Visualizer()
    
    errors_dict = {
        'Rectangle (ordre 1)': errors_rect,
        'Trapèzes (ordre 2)': errors_trap,
        'Simpson 1/3 (ordre 4)': errors_simp,
        'Simpson 3/8 (ordre 4)': errors_s38
    }
    
    fig = visualizer.plot_convergence(
        n_values,
        errors_dict,
        title="Convergence des méthodes d'intégration - f(x)=e^x sur [0,1]",
        xlabel="Nombre de subdivisions n",
        ylabel="Erreur absolue"
    )
    
    # Sauvegarder le graphique
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    fig.savefig(os.path.join(results_dir, "convergence_integration.png"), dpi=150)
    print(f"Graphique sauvegardé: results/convergence_integration.png")
    plt.close(fig)
    
    print()
    
    # Comparaison adaptive vs Simpson composé (n=100)
    print_subheader("Comparaison: Simpson adaptatif vs Simpson composé (n=100)")
    
    # Simpson composé avec n=100
    simp_100 = NewtonCotes.simpson(f, a, b, 100)
    err_simp_100 = abs(simp_100 - exact)
    
    # Simpson adaptatif avec différentes tolérances
    print(f"{'Tolérance':>14}  {'Résultat':>18}  {'Erreur':>14}  {'Évaluations':>12}")
    print_separator("-")
    
    for tol in [1e-2, 1e-4, 1e-6, 1e-8]:
        integrator = AdaptiveIntegration(tol=tol)
        result = integrator.adaptive_simpson(f, a, b)
        err = abs(result - exact)
        print(f"{tol:>14.2e}  {result:>18.12f}  {err:>14.2e}  {'variable':>12}")
    
    print(f"{'Simpson n=100':>14}  {simp_100:>18.12f}  {err_simp_100:>14.2e}  {'101':>12}")
    
    print()
    print("Analyse: L'intégration adaptative ajuste automatiquement le nombre")
    print("         de subdivisions pour atteindre la tolérance souhaitée.")
    print("         Avantage: précision ciblée avec moins d'évaluations.")
    
    return n_values, errors_dict


# ─────────────────────────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────────────────────────

def main():
    """
    Point d'entrée du programme.

    Lance toutes les études de convergence pour les méthodes
    d'intégration numérique implémentées.
    """
    print_header("ÉTUDE DE CONVERGENCE DES MÉTHODES D'INTÉGRATION NUMÉRIQUE")

    # ── 1. Méthodes de Newton-Cotes ──────────────────────────────
    print_header("1. MÉTHODES DE NEWTON-COTES — Erreurs en fonction de n")
    for func_info in TEST_FUNCTIONS:
        study_newton_cotes(func_info)
        print()

    # ── 2. Intégration adaptative ────────────────────────────────
    print_header("2. INTÉGRATION ADAPTATIVE (Simpson) — Précision vs tolérance")
    for func_info in TEST_FUNCTIONS:
        study_adaptive(func_info)
        print()

    # ── 3. Quadrature de Gauss (bonus) ───────────────────────────
    print_header("3. QUADRATURE DE GAUSS-LEGENDRE (BONUS) — Comparaison globale")
    for func_info in TEST_FUNCTIONS:
        study_gauss(func_info)
        print()

    # ── 4. Ordres de convergence empiriques ──────────────────────
    print_header("4. ORDRES DE CONVERGENCE EMPIRIQUES")
    for func_info in TEST_FUNCTIONS:
        estimate_convergence_order(func_info)
        print()

    # ── 5. Analyse complète (section 4.2) ────────────────────────
    print_header("5. ANALYSE COMPLÈTE - SECTION 4.2 DU PROJET")
    analyze_integration_convergence()
    print()

    print_separator("=")
    print("  Fin du programme.")
    print_separator("=")


if __name__ == "__main__":
    main()