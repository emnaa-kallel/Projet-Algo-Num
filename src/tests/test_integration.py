import numpy as np
import sys
import os

# 🔥 Permet d'importer les modules du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integration.newton_cotes import NewtonCotes
from visualization.visualizer import Visualizer


def test_integration_methods():
    # Fonction test
    f = lambda x: np.exp(x)
    exact = np.e - 1

    # valeurs de n
    n_values = [2, 4, 8, 16, 32, 64]

    # méthodes
    methods = {
        "Rectangle": NewtonCotes.rectangle,
        "Trapèze": NewtonCotes.trapezoidal,
        "Simpson": NewtonCotes.simpson
    }

    viz = Visualizer()

    fig = viz.plot_integration_comparison(
        f,
        0,
        1,
        n_values,
        methods,
        exact,
        "Convergence des méthodes d'intégration"
    )

    # 🔥 sauvegarde
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/integration_convergence.png", dpi=300)

    print("✅ Test terminé : graphique généré !")


if __name__ == "__main__":
    test_integration_methods()