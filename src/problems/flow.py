import numpy as np
import pandas as pd

from src.interpolation.polynomial import PolynomialInterpolation
from src.integration import NewtonCotes, AdaptiveIntegration


class FlowProblem:
    """
    Problème d'écoulement dans un canal.

    Débit total : D = ∫[0,6] v(x)·w(x) dx
    Largeur     : w(x) = 0.5 + 0.1·x  (m)

    Paramètres
    ----------
    x_data     : array    — positions de mesure (m)
    v_data     : array    — vitesses mesurées (m/s)
    width_func : callable — fonction largeur w(x) (défaut : 0.5 + 0.1x)
    """

    def __init__(self, x_data, v_data, width_func=None):
        self.x_data = np.asarray(x_data, dtype=float)
        self.v_data = np.asarray(v_data, dtype=float)
        self.width_func = width_func if width_func is not None else (lambda x: 0.5 + 0.1 * x)
        self._interpolator = PolynomialInterpolation(self.x_data, self.v_data)

    # ─── Vitesse interpolée ───────────────────────────────────────────────────

    def velocity(self, x_eval, method='newton'):
        """
        Retourne la vitesse interpolée en x_eval.

        Paramètres
        ----------
        x_eval : float ou array
        method : 'newton' (défaut) ou 'lagrange'

        Retourne
        --------
        float ou array — v(x) en m/s
        """
        return self._interpolator.evaluate(x_eval, method=method)

    # ─── Débit local ──────────────────────────────────────────────────────────

    def local_flow_rate(self, x_eval, method='newton'):
        """
        Débit élémentaire : q(x) = v(x) * w(x).

        Paramètres
        ----------
        x_eval : float ou array
        method : 'newton' ou 'lagrange'

        Retourne
        --------
        float ou array — q(x) en m²/s
        """
        v = self.velocity(x_eval, method=method)
        w = self.width_func(np.asarray(x_eval, dtype=float))
        return v * w

    # ─── Débit total ──────────────────────────────────────────────────────────

    def total_flow_rate(self, method='adaptive', n=100):
        """
        Calcule D = ∫[0,6] v(x)·w(x) dx.

        Paramètres
        ----------
        method : str — 'adaptive' (défaut), 'trapeze', 'simpson'
        n      : int — sous-intervalles pour les méthodes composées

        Retourne
        --------
        float — débit volumique total D (m³/s)
        """
        a, b = self.x_data[0], self.x_data[-1]

        if method == 'trapeze':
            # Trapèzes sur n sous-intervalles via l'interpolateur
            f = lambda x: self._interpolator.evaluate(x, 'newton') * self.width_func(x)
            return NewtonCotes.trapezoidal(f, a, b, n)

        elif method == 'simpson':
            # Simpson composé sur n sous-intervalles via l'interpolateur
            f = lambda x: self._interpolator.evaluate(x, 'newton') * self.width_func(x)
            return NewtonCotes.simpson(f, a, b, n if n % 2 == 0 else n + 1)

        elif method == 'adaptive':
            ai = AdaptiveIntegration(tol=1e-6)
            f = lambda x: self._interpolator.evaluate(x, 'newton') * self.width_func(x)
            return ai.adaptive_simpson(f, a, b)

        else:
            raise ValueError(f"Méthode inconnue : '{method}'. Choisir 'adaptive', 'trapeze' ou 'simpson'.")

    # ─── Accélération ─────────────────────────────────────────────────────────

    def acceleration(self, x_eval):
        """
        Approximation de l'accélération dv/dx par différences finies
        sur un tableau de points x_eval (interpolés).

        Paramètres
        ----------
        x_eval : array — positions où évaluer dv/dx

        Retourne
        --------
        array — dv/dx (m/s par m)
        """
        x_eval = np.asarray(x_eval, dtype=float)
        v_eval = self._interpolator.evaluate(x_eval, 'newton')
        return np.gradient(v_eval, x_eval)

    # ─── Travail ──────────────────────────────────────────────────────────────

    def work(self, mass=2.0):
        """
        Calcule le travail exercé sur une particule de masse `mass`
        traversant le canal :

            W = ∫[0,6] F(x) dx = ∫[0,6] mass * v(x) * (dv/dx) dx

        L'intégrale est évaluée numériquement par la méthode adaptative
        de Simpson, en approchant dv/dx par différences finies.

        Paramètres
        ----------
        mass : float — masse de la particule en kg (défaut 2.0)

        Retourne
        --------
        float — travail W (J)
        """
        # Grille fine pour approximer dv/dx par différences finies
        x_fine = np.linspace(self.x_data[0], self.x_data[-1], 500)
        v_fine = self._interpolator.evaluate(x_fine, 'newton')
        dvdx_fine = np.gradient(v_fine, x_fine)

        # Interpolation de dv/dx pour l'intégration adaptative (np uniquement)
        dvdx_interp = lambda x: np.interp(x, x_fine, dvdx_fine)

        ai = AdaptiveIntegration(tol=1e-6)
        f = lambda x: mass * self._interpolator.evaluate(x, 'newton') * dvdx_interp(x)
        return ai.adaptive_simpson(f, self.x_data[0], self.x_data[-1])

    # ─── Rapport terminal ─────────────────────────────────────────────────────

    def report(self):
        """Affiche un résumé complet dans le terminal."""
        print("=" * 60)
        print("  PROBLÈME D'ÉCOULEMENT")
        print("=" * 60)

        for x_q in [1.5, 3.5, 5.0]:
            v_lag = self.velocity(x_q, 'lagrange')
            v_new = self.velocity(x_q, 'newton')
            print(f"\n  v({x_q}) — Lagrange  : {v_lag:.4f} m/s")
            print(f"  v({x_q}) — Newton    : {v_new:.4f} m/s")

        print("\n  Débit total D :")
        for m in ['trapeze', 'simpson', 'adaptive']:
            D = self.total_flow_rate(m)
            print(f"    {m:10s} : {D:.4f} m³/s")

        print("\n  Travail (masse = 2 kg) :")
        W = self.work(mass=2.0)
        print(f"    W = {W:.4f} J")
        print("=" * 60)