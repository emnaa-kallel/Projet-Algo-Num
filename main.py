import numpy as np

# Import de tes classes (à adapter selon ton arborescence)
from src.interpolation.polynomial import PolynomialInterpolation
from src.visualization.visualizer import Visualizer


# =========================
# 1. DONNÉES EXPÉRIMENTALES
# =========================
def load_data():
    """
    Charge les données du projet.
    """
    # Refroidissement
    t_data = np.array([0,1,2,3,4,5,6,7,8,9,10])
    T_data = np.array([90,85,72,63,58,52,48,45,43,41,40])

    # Écoulement
    x_data = np.array([0,0.5,1.2,1.8,2.5,3.1,3.7,4.2,4.8,5.3,6.0])
    v_data = np.array([0,2.1,3.8,5.2,6.4,7.0,7.3,7.2,6.8,5.9,4.5])

    return t_data, T_data, x_data, v_data


# largeur canal
def width_function(x):
    return 0.5 + 0.1 * x


# =========================
# MAIN
# =========================
def main():

    vis = Visualizer()

    # =========================
    # 1. LOAD DATA
    # =========================
    t_data, T_data, x_data, v_data = load_data()

    t_fine = np.linspace(0, 10, 200)
    x_fine = np.linspace(0, 6, 200)

    # =========================
    # 2. RUNGE PHENOMENON
    # =========================
    runge_x = np.linspace(-1, 1, 200)
    runge_true = 1 / (1 + 25 * runge_x**2)

    # interpolation test
    x_nodes = np.linspace(-1, 1, 10)
    y_nodes = 1 / (1 + 25 * x_nodes**2)

    interp = PolynomialInterpolation(x_nodes, y_nodes)

    runge_interp = interp.lagrange(runge_x)

    vis.plot_runge_phenomenon(
        runge_x,
        runge_true,
        {"Lagrange": runge_interp}
    )


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()