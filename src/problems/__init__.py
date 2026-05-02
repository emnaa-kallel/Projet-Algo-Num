"""
Module de problèmes d'analyse numérique.

Ce package contient les classes pour différents problèmes :
- Refroidissement d'un composant
- Écoulement de fluide
"""

from .cooling import CoolingProblem
from .flow import FlowProblem

__all__ = [
    "CoolingProblem",
    "FlowProblem",
]
