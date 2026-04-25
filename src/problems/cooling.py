import numpy as np
import pandas as pd

from src.interpolation.polynomial import PolynomialInterpolation
from src.integration import NewtonCotes, AdaptiveIntegration


class CoolingProblem:
    """
    Problème de refroidissement d'un composant électronique.

    Modèle physique :
        T(t) = T_amb + (T0 - T_amb) * exp(-k*t)

    Q = ∫[0,10] h * (T(t) - T_amb) dt
    avec h = 50 J/°C et T_amb = 20 °C (valeurs du sujet).

    Paramètres
    ----------
    t_data      : array — instants de mesure
    T_data      : array — températures mesurées
    T_ambient   : float — température ambiante (défaut 20.0 °C)
    h_coeff     : float — coefficient de convection (défaut 50.0 J/°C)
    """

    def __init__(self, t_data, T_data, T_ambient=20.0, h_coeff=50.0):
        self.t_data = np.asarray(t_data, dtype=float)
        self.T_data = np.asarray(T_data, dtype=float)
        self.T_ambient = T_ambient
        self.h_coeff = h_coeff
        self._interpolator = PolynomialInterpolation(self.t_data, self.T_data)

    # ─── Interpolation ────────────────────────────────────────────────────────

    def temperature(self, t_eval, method='newton'):
        """
        Retourne la température interpolée à l'instant t_eval.

        Paramètres
        ----------
        t_eval : float ou array
        method : 'newton' (défaut) ou 'lagrange'

        Retourne
        --------
        float ou array — T interpolée (°C)
        """
        return self._interpolator.evaluate(t_eval, method=method)

    # ─── Taux de perte de chaleur ─────────────────────────────────────────────

    def heat_loss_rate(self, t_eval, method='newton'):
        """
        Taux de perte de chaleur instantané : q(t) = h * (T(t) - T_amb).

        Paramètres
        ----------
        t_eval : float ou array
        method : 'newton' ou 'lagrange'

        Retourne
        --------
        float ou array — q(t) en W/m²
        """
        T = self.temperature(t_eval, method=method)
        return self.h_coeff * (T - self.T_ambient)

    # ─── Intégration : chaleur totale dissipée ────────────────────────────────

    def total_heat_loss(self, method='adaptive', n=100):
        """
        Calcule Q = ∫[0,10] h*(T(t)-T_amb) dt.

        Paramètres
        ----------
        method : str — 'adaptive' (défaut), 'trapeze', 'simpson'
        n      : int — nombre de sous-intervalles pour les méthodes composées

        Retourne
        --------
        float — chaleur totale dissipée Q (J/m²)
        """
        a, b = self.t_data[0], self.t_data[-1]

        if method == 'trapeze':
            # Trapèzes sur les données discrètes via NewtonCotes.trapezoidal
            f = lambda t: self.h_coeff * (self._interpolator.evaluate(t, 'newton') - self.T_ambient)
            return NewtonCotes.trapezoidal(f, a, b, n)

        elif method == 'simpson':
            # Simpson composé sur n sous-intervalles via l'interpolateur
            f = lambda t: self.h_coeff * (self._interpolator.evaluate(t, 'newton') - self.T_ambient)
            return NewtonCotes.simpson(f, a, b, n if n % 2 == 0 else n + 1)

        elif method == 'adaptive':
            ai = AdaptiveIntegration(tol=1e-6)
            f = lambda t: self.h_coeff * (self._interpolator.evaluate(t, 'newton') - self.T_ambient)
            return ai.adaptive_simpson(f, a, b)

        else:
            raise ValueError(f"Méthode inconnue : '{method}'. Choisir 'adaptive', 'trapeze' ou 'simpson'.")

    # ─── Modèle exponentiel ───────────────────────────────────────────────────

    def exponential_model(self, t_eval, k):
        """
        Évalue le modèle exponentiel T(t) = T_amb + (T0 - T_amb)*exp(-k*t).

        Paramètres
        ----------
        t_eval : float ou array — instants d'évaluation
        k      : float          — constante de refroidissement

        Retourne
        --------
        float ou array — températures modélisées (°C)
        """
        T0 = self.T_data[0]
        return self.T_ambient + (T0 - self.T_ambient) * np.exp(-k * np.asarray(t_eval, dtype=float))

    # ─── Erreur du modèle ─────────────────────────────────────────────────────

    def model_error(self, k):
        """
        Calcule E(k) = ∫[0,10] |T_exp(t) - T_modele(t,k)| dt
        par intégration adaptative.

        Paramètres
        ----------
        k : float — constante à tester

        Retourne
        --------
        float — erreur intégrée E(k)
        """
        ai = AdaptiveIntegration(tol=1e-4)
        f = lambda t: abs(
            self._interpolator.evaluate(t, 'newton') - self.exponential_model(t, k)
        )
        return ai.adaptive_simpson(f, self.t_data[0], self.t_data[-1])

    # ─── Estimation de k par bissection ──────────────────────────────────────

    def estimate_k(self, k_min=0.01, k_max=0.5, tol=1e-4):
        """
        Estime k optimal en minimisant E(k) par bissection sur la dérivée.

        La fonction E(k) est convexe : on cherche k* tel que dE/dk = 0
        par la méthode de bissection classique.

        Paramètres
        ----------
        k_min : float — borne inférieure pour k (défaut 0.01)
        k_max : float — borne supérieure pour k (défaut 0.5)
        tol   : float — tolérance sur k (défaut 1e-4)

        Retourne
        --------
        float — valeur optimale de k
        """
        # Recherche grossière sur grille pour encadrer le minimum
        k_grid = np.linspace(k_min, k_max, 50)
        errors = [self.model_error(k) for k in k_grid]
        idx = np.argmin(errors)

        # Encadrement autour du minimum trouvé
        a = k_grid[max(0, idx - 1)]
        b = k_grid[min(len(k_grid) - 1, idx + 1)]

        # Bissection sur la dérivée numérique de E(k)
        # E est convexe → dE/dk = 0 au minimum
        eps = 1e-6
        while b - a > tol:
            mid = (a + b) / 2.0
            dE = (self.model_error(mid + eps) - self.model_error(mid - eps)) / (2 * eps)
            if dE > 0:
                b = mid
            else:
                a = mid

        return (a + b) / 2.0

    # ─── Rapport terminal ─────────────────────────────────────────────────────

    def report(self):
        """Affiche un résumé complet dans le terminal."""
        print("=" * 60)
        print("  PROBLÈME DE REFROIDISSEMENT")
        print("=" * 60)

        for t_q in [2.5, 7.3]:
            T_lag = self.temperature(t_q, 'lagrange')
            T_new = self.temperature(t_q, 'newton')
            print(f"\n  T({t_q}) — Lagrange  : {T_lag:.4f} °C")
            print(f"  T({t_q}) — Newton    : {T_new:.4f} °C")

        print("\n  Chaleur dissipée Q :")
        for m in ['trapeze', 'simpson', 'adaptive']:
            Q = self.total_heat_loss(m)
            print(f"    {m:10s} : {Q:.2f} J/m²")

        print("\n  Estimation de k :")
        k_opt = self.estimate_k()
        print(f"    k optimal = {k_opt:.4f} s⁻¹")
        print(f"    E(k_opt)  = {self.model_error(k_opt):.4f}")
        print("=" * 60)