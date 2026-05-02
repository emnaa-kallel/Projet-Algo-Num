import numpy as np
import matplotlib.pyplot as plt
import os


class Visualizer:
    """
    Classe de visualisation pour les méthodes d'analyse numérique :
    interpolation, convergence, Runge, refroidissement et écoulement.
    """

    def __init__(self, style="seaborn-v0_8-darkgrid", figsize=(10, 6), results_dir="results"):
        """
        Initialise le visualiseur.

        Parameters
        ----------
        style : str
            Style matplotlib utilisé pour les graphiques.
        figsize : tuple
            Taille des figures.
        results_dir : str
            Répertoire où sauvegarder les figures.
        """
        try:
         plt.style.use(style)
        except:
            plt.style.use("default")
        self.figsize = figsize
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    # =========================
    # INTERPOLATION
    # =========================
    def plot_interpolation_comparison(self, x_data, y_data, interpolators, x_fine, title):
        """
        Compare plusieurs méthodes d'interpolation sur un même graphique.

        Parameters
        ----------
        x_data : array-like
            Points expérimentaux en x.
        y_data : array-like
            Valeurs expérimentales en y.
        interpolators : dict
            Dictionnaire {nom: interpolateur}.
        x_fine : array-like
            Points fins pour tracer les courbes.
        title : str
            Titre du graphique.
        """
        plt.figure(figsize=self.figsize)

        plt.scatter(x_data, y_data, color="black", label="Données")

        for name, interp in interpolators.items():
            y_fine = interp.evaluate(x_fine)
            plt.plot(x_fine, y_fine, label=name)

        plt.title(title)
        plt.legend()
        plt.grid()
        filepath = os.path.join(self.results_dir, "interpolation_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    # =========================
    # RUNGE
    # =========================
    def plot_runge_phenomenon(self, x_fine, y_true, interpolations):
        """
        Visualise le phénomène de Runge en comparant plusieurs interpolations.

        Parameters
        ----------
        x_fine : array-like
            Domaine continu.
        y_true : array-like
            Fonction exacte.
        interpolations : dict
            {nom: valeurs interpolées}.
        """
        plt.figure(figsize=self.figsize)

        plt.plot(x_fine, y_true, "k--", label="Fonction exacte")

        for name, y_interp in interpolations.items():
            plt.plot(x_fine, y_interp, label=name)

        plt.title("Phénomène de Runge")
        plt.legend()
        plt.grid()
        filepath = os.path.join(self.results_dir, "runge_phenomenon.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    # =========================
    # CONVERGENCE
    # =========================
    def plot_convergence(self, n_values, errors, methods, title):
        """
        Trace l'évolution de l'erreur en fonction du nombre de points.

        Parameters
        ----------
        n_values : array-like
            Nombre de subdivisions.
        errors : list of array-like
            Erreurs pour chaque méthode.
        methods : list of str
            Noms des méthodes.
        title : str
            Titre du graphique.
        """
        plt.figure(figsize=self.figsize)

        for i, method in enumerate(methods):
            plt.plot(n_values, errors[i], marker="o", label=method)

        plt.xscale("log")
        plt.yscale("log")

        plt.title(title)
        plt.xlabel("Nombre de points (n)")
        plt.ylabel("Erreur")
        plt.legend()
        plt.grid(True, which="both")
        filepath = os.path.join(self.results_dir, "convergence.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    # =========================
    # REFROIDISSEMENT
    # =========================
    def plot_cooling_analysis(self, t_data, T_data, t_fine, T_interp, k_opt, T_model):
        """
        Analyse graphique du problème de refroidissement.

        Parameters
        ----------
        t_data : array-like
            Temps expérimentaux.
        T_data : array-like
            Températures mesurées.
        t_fine : array-like
            Temps continu pour affichage.
        T_interp : array-like
            Interpolation des données.
        k_opt : float
            Paramètre optimal du modèle exponentiel.
        T_model : array-like
            Modèle physique.
        """
        plt.figure(figsize=self.figsize)

        plt.scatter(t_data, T_data, color="black", label="Données")
        plt.plot(t_fine, T_interp, label="Interpolation")
        plt.plot(t_fine, T_model, label=f"Modèle (k={k_opt:.4f})")

        plt.title("Analyse du refroidissement")
        plt.xlabel("Temps (s)")
        plt.ylabel("Température (°C)")
        plt.legend()
        plt.grid()
        filepath = os.path.join(self.results_dir, "cooling_analysis.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    # =========================
    # ÉCOULEMENT
    # =========================
    def plot_flow_analysis(self, x_data, v_data, x_fine, v_interp, w_function):
        """
        Analyse graphique de l'écoulement dans un canal.

        Parameters
        ----------
        x_data : array-like
            Positions mesurées.
        v_data : array-like
            Vitesses mesurées.
        x_fine : array-like
            Domaine continu.
        v_interp : array-like
            Interpolation de la vitesse.
        w_function : callable
            Fonction largeur du canal w(x).
        """
        plt.figure(figsize=self.figsize)

        plt.scatter(x_data, v_data, color="black", label="Données vitesse")
        plt.plot(x_fine, v_interp, label="Interpolation vitesse")

        plt.plot(x_fine, w_function(x_fine), label="Largeur du canal w(x)")

        plt.title("Analyse de l'écoulement")
        plt.xlabel("x (m)")
        plt.ylabel("Valeurs")
        plt.legend()
        plt.grid()
        filepath = os.path.join(self.results_dir, "flow_analysis.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()