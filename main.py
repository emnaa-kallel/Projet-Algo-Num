import numpy as np

from src.interpolation.polynomial import PolynomialInterpolation
from src.visualization.visualizer import Visualizer


# =========================
# 1. DONNÉES EXPÉRIMENTALES
# =========================
def load_data():
    t_data = np.array([0,1,2,3,4,5,6,7,8,9,10])
    T_data = np.array([90,85,72,63,58,52,48,45,43,41,40])

    x_data = np.array([0,0.5,1.2,1.8,2.5,3.1,3.7,4.2,4.8,5.3,6.0])
    v_data = np.array([0,2.1,3.8,5.2,6.4,7.0,7.3,7.2,6.8,5.9,4.5])

    return t_data, T_data, x_data, v_data


def width_function(x):
    return 0.5 + 0.1 * x


# =========================
# MAIN
# =========================
def main():

    vis = Visualizer()

    # =========================
    # LOAD DATA
    # =========================
    t_data, T_data, x_data, v_data = load_data()

    t_fine = np.linspace(0, 10, 200)
    x_fine = np.linspace(0, 6, 200)

    # =========================
    # RUNGE PHENOMENON
    # =========================
    runge_x = np.linspace(-1, 1, 200)
    runge_true = 1 / (1 + 25 * runge_x**2)

    # Equidistant points
    n = 10
    x_eq = np.linspace(-1, 1, n)
    y_eq = 1 / (1 + 25 * x_eq**2)

    interp_eq = PolynomialInterpolation(x_eq, y_eq)
    runge_eq = interp_eq.lagrange(runge_x)

    # Chebyshev points
    k = np.arange(n)
    x_cheb = np.cos((2*k + 1) / (2*n) * np.pi)
    x_cheb = np.sort(x_cheb)

    y_cheb = 1 / (1 + 25 * x_cheb**2)

    interp_cheb = PolynomialInterpolation(x_cheb, y_cheb)
    runge_cheb = interp_cheb.lagrange(runge_x)

    # Plot
    vis.plot_runge_phenomenon(
        runge_x,
        runge_true,
        {
            "Équidistant": runge_eq,
            "Tchebychev": runge_cheb
        }
    )


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()