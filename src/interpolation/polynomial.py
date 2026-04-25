import numpy as np


class PolynomialInterpolation:
    """
    Classe d'interpolation polynomiale :
    - Lagrange
    - Newton (différences divisées)
    """

    def __init__(self, x_points, y_points):
        """
        Initialise l'interpolateur.

        Parameters
        ----------
        x_points : array-like
            Points x connus.
        y_points : array-like
            Valeurs y correspondantes.

        Raises
        ------
        ValueError
            Si les tailles sont différentes ou si x contient des doublons.
        """
        self.x = np.array(x_points, dtype=float)
        self.y = np.array(y_points, dtype=float)

        if len(self.x) != len(self.y):
            raise ValueError("x_points et y_points doivent avoir la même taille.")

        if len(np.unique(self.x)) != len(self.x):
            raise ValueError("Les points x doivent être distincts.")

        self.n = len(self.x)
        self._newton_coeffs = None

    # =========================
    # LAGRANGE
    # =========================
    def lagrange(self, x_eval):
        """
        Évalue le polynôme d'interpolation de Lagrange.

        Parameters
        ----------
        x_eval : float ou array-like

        Returns
        -------
        float ou np.ndarray
        """
        x_eval = np.array(x_eval, dtype=float)

        result = np.zeros_like(x_eval, dtype=float)

        for i in range(self.n):
            Li = np.ones_like(x_eval)

            for j in range(self.n):
                if i != j:
                    Li *= (x_eval - self.x[j]) / (self.x[i] - self.x[j])

            result += self.y[i] * Li

        return result

    # =========================
    # NEWTON COEFFICIENTS
    # =========================
    def newton_coefficients(self):
        """
        Calcule les coefficients des différences divisées (Newton).

        Returns
        -------
        np.ndarray
            coefficients du polynôme de Newton
        """
        n = self.n
        coef = np.copy(self.y).astype(float)

        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                coef[i] = (coef[i] - coef[i - 1]) / (self.x[i] - self.x[i - j])

        self._newton_coeffs = coef
        return coef

    # =========================
    # NEWTON EVALUATION
    # =========================
    def newton_eval(self, x_eval):
        """
        Évalue le polynôme de Newton.

        Parameters
        ----------
        x_eval : float ou array-like

        Returns
        -------
        float ou np.ndarray
        """
        if self._newton_coeffs is None:
            self.newton_coefficients()

        x_eval = np.array(x_eval, dtype=float)
        result = np.zeros_like(x_eval, dtype=float)

        for i in range(self.n - 1, -1, -1):
            result = result * (x_eval - self.x[i]) + self._newton_coeffs[i]

        return result

    # =========================
    # INTERFACE UNIFIÉE
    # =========================
    def evaluate(self, x_eval, method="newton"):
        """
        Interface d'évaluation.

        Parameters
        ----------
        x_eval : float ou array-like
        method : str
            'newton' ou 'lagrange'

        Returns
        -------
        np.ndarray
        """
        if method == "newton":
            return self.newton_eval(x_eval)
        elif method == "lagrange":
            return self.lagrange(x_eval)
        else:
            raise ValueError("Méthode inconnue. Utiliser 'newton' ou 'lagrange'.")