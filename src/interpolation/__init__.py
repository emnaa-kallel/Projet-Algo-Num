"""
Module d'interpolation polynomiale.

Ce package regroupe les méthodes d'interpolation :
- Lagrange
- Newton (différences divisées)
- Splines (à implémenter)
"""

from .polynomial import PolynomialInterpolation

__all__ = [
    "PolynomialInterpolation",
]
