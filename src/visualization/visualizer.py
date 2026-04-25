import numpy as np
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, style="seaborn-v0_8-darkgrid", figsize=(10, 6)):
        self.style = style
        self.figsize = figsize
        self._COLORS = ["#E63946", "#2196F3", "#4CAF50", "#FF9800"]

    def plot_integration_comparison(
        self,
        f,
        a,
        b,
        n_values,
        methods_dict,
        exact_value,
        title="Comparaison des méthodes d'intégration"
    ):

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            for idx, (name, method) in enumerate(methods_dict.items()):
                color = self._COLORS[idx % len(self._COLORS)]

                errors = []

                for n in n_values:
                    approx = method(f, a, b, n)
                    errors.append(abs(approx - exact_value))

                ax.loglog(
                    n_values,
                    errors,
                    marker='o',
                    linewidth=2,
                    label=name,
                    color=color
                )

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("n (subdivisions)")
            ax.set_ylabel("Erreur absolue")
            ax.legend()
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

            fig.tight_layout()

        return fig

    def plot_convergence(
        self,
        n_values,
        errors_dict,
        title="Étude de convergence - Erreur en fonction de n",
        xlabel="Nombre de subdivisions n",
        ylabel="Erreur absolue"
    ):
        """
        Trace les courbes de convergence en échelle log-log.
        
        Cette méthode permet de visualiser la vitesse de convergence
        des différentes méthodes d'intégration numérique.
        
        Paramètres
        ----------
        n_values : list ou array
            Liste des valeurs de n (nombre de subdivisions).
        errors_dict : dict
            Dictionnaire {nom_méthode: [erreurs pour chaque n]}.
        title : str
            Titre du graphique.
        xlabel, ylabel : str
            Labels des axes.
            
        Retourne
        -------
        fig : matplotlib.figure.Figure
            Figure matplotlib créée.
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            for idx, (name, errors) in enumerate(errors_dict.items()):
                color = self._COLORS[idx % len(self._COLORS)]
                ax.loglog(
                    n_values,
                    errors,
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=name,
                    color=color
                )

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

            # Ajuster les ticks pour une meilleure lisibilité
            ax.tick_params(axis='both', which='major', labelsize=10)

            fig.tight_layout()

        return fig
    