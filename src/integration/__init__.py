"""
Module d'intégration numérique.

Ce package regroupe les méthodes d'intégration :
- Newton-Cotes (rectangle, trapèze, Simpson)
- Simpson adaptatif
- (optionnel) Quadrature de Gauss
"""

from .newton_cotes import NewtonCotes
from .adaptive_integration import AdaptiveIntegration

# Bonus (si implémenté)
try:
    from .gauss_quadrature import GaussQuadrature
except ImportError:
    GaussQuadrature = None

__all__ = [
    "NewtonCotes",
    "AdaptiveIntegration",
    "GaussQuadrature",
]