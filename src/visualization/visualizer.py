import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

COLORS = {
    "lagrange":  "#2196F3",
    "newton":    "#FF5722",
    "data":      "#4CAF50",
    "model":     "#9C27B0",
    "uniform":   "#F44336",
    "chebyshev": "#00BCD4",
    "vx":        "#00BCD4",
    "vwx":       "#FF9800",
}

_PALETTE = ["#E63946", "#2196F3", "#4CAF50", "#FF9800"]


class Visualizer:
    """
    Classe de visualisation du projet d'analyse numérique.

    Paramètres
    ----------
    style   : str   — style matplotlib (défaut 'seaborn-v0_8-darkgrid')
    figsize : tuple — taille par défaut des figures (défaut (10, 6))
    """

    def __init__(self, style='seaborn-v0_8-darkgrid', figsize=(10, 6)):
        self.figsize = figsize
        self.style = style
        self._COLORS = _PALETTE
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('default')

    # ─── 1. Comparaison interpolations ────────────────────────────────────────

    def plot_interpolation_comparison(self, x_data, y_data, interpolators, x_fine, title=""):
        """
        Trace plusieurs interpolations sur le même graphe.

        Paramètres
        ----------
        x_data        : array — nœuds (données réelles)
        y_data        : array — valeurs aux nœuds
        interpolators : dict  — {"nom": array y_interp sur x_fine}
        x_fine        : array — points fins pour tracé
        title         : str   — titre du graphe

        Retourne
        --------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(x_data, y_data, color=COLORS["data"], zorder=6, s=60, label="Données")
        palette = [COLORS["lagrange"], COLORS["newton"], "#FF9800", "#9C27B0"]
        for (name, y_interp), color in zip(interpolators.items(), palette):
            ax.plot(x_fine, y_interp, color=color, lw=2, label=name)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        return fig

    # ─── 2. Comparaison méthodes d'intégration (ajout branche ami) ───────────

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
        """
        Trace les courbes d'erreur en log-log pour plusieurs méthodes
        d'intégration en faisant varier n.

        Paramètres
        ----------
        f           : callable — fonction à intégrer
        a, b        : float    — bornes d'intégration
        n_values    : list     — valeurs de n testées
        methods_dict: dict     — {"nom": callable(f, a, b, n)}
        exact_value : float    — valeur exacte de l'intégrale
        title       : str      — titre du graphe

        Retourne
        --------
        matplotlib.figure.Figure
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            for idx, (name, method) in enumerate(methods_dict.items()):
                color = self._COLORS[idx % len(self._COLORS)]
                errors = [abs(method(f, a, b, n) - exact_value) for n in n_values]
                ax.loglog(n_values, errors, marker='o', linewidth=2, label=name, color=color)

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("n (subdivisions)")
            ax.set_ylabel("Erreur absolue")
            ax.legend()
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            fig.tight_layout()

        return fig

    # ─── 3. Phénomène de Runge ────────────────────────────────────────────────

    def plot_runge_phenomenon(self, x_fine, y_true, interpolations):
        """
        Compare nœuds équidistants vs Tchebychev pour chaque n.

        Paramètres
        ----------
        x_fine         : array — points fins
        y_true         : array — valeurs exactes
        interpolations : dict  — résultat de demonstrate_runge()
                         {"uniform": {n: {...}}, "chebyshev": {n: {...}}}

        Retourne
        --------
        matplotlib.figure.Figure
        """
        n_list = list(interpolations["uniform"].keys())
        fig, axes = plt.subplots(2, len(n_list), figsize=(5 * len(n_list), 8))
        if len(n_list) == 1:
            axes = axes.reshape(2, 1)

        for col, n in enumerate(n_list):
            for row, (key, color, label) in enumerate([
                ("uniform",   COLORS["uniform"],   "Équidistants"),
                ("chebyshev", COLORS["chebyshev"], "Tchebychev"),
            ]):
                ax = axes[row][col]
                poly = interpolations[key][n]
                ax.plot(x_fine, y_true, color=COLORS["data"], lw=2, label="f(x) réelle")
                ax.plot(x_fine, poly["y_interp"], color=color, lw=2, label=f"{label} n={n}")
                ax.scatter(poly["x_nodes"], poly["y_nodes"], color="white", zorder=5, s=40)
                ax.set_ylim(-1.5, 1.5)
                ax.set_title(f"{label} — n={n}\nErreur max={poly['max_error']:.3f}")
                ax.legend(fontsize=7)
                ax.grid(True)

        fig.suptitle("Phénomène de Runge — 1/(1+25x²)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    # ─── 4. Convergence ───────────────────────────────────────────────────────

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

        Paramètres
        ----------
        n_values   : list ou array — valeurs de n
        errors_dict: dict          — {"nom_méthode": [erreurs pour chaque n]}
        title      : str           — titre du graphique
        xlabel     : str           — label axe x
        ylabel     : str           — label axe y

        Retourne
        --------
        matplotlib.figure.Figure
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            for idx, (name, errors) in enumerate(errors_dict.items()):
                color = self._COLORS[idx % len(self._COLORS)]
                ax.loglog(n_values, errors, marker='o', linewidth=2,
                          markersize=8, label=name, color=color)

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.tick_params(axis='both', which='major', labelsize=10)
            fig.tight_layout()

        return fig

    # ─── 5. Analyse refroidissement ───────────────────────────────────────────

    def plot_cooling_analysis(self, t_data, T_data, t_fine, T_interp, k_opt, T_model):
        """
        Graphe complet du refroidissement :
        données + interpolation Newton + modèle exponentiel.

        Paramètres
        ----------
        t_data   : array — temps mesurés
        T_data   : array — températures mesurées
        t_fine   : array — temps fins pour tracé
        T_interp : array — températures interpolées sur t_fine
        k_opt    : float — constante k optimale
        T_model  : array — températures du modèle exp sur t_fine

        Retourne
        --------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(t_data, T_data, color=COLORS["data"], zorder=6, s=70, label="Données mesurées")
        ax.plot(t_fine, T_interp, color=COLORS["newton"], lw=2, label="Interpolation Newton")
        ax.plot(t_fine, T_model, color=COLORS["model"], lw=2, linestyle="--",
                label=f"Modèle exp (k={k_opt:.4f})")
        ax.axhline(20.0, color="#555", linestyle="-.", lw=1, label="T_amb = 20°C")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Température (°C)")
        ax.set_title("Refroidissement du composant électronique")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        return fig

    # ─── 6. Analyse écoulement ────────────────────────────────────────────────

    def plot_flow_analysis(self, x_data, v_data, x_fine, v_interp, w_function):
        """
        Deux sous-graphes : v(x) et v(x)·w(x).

        Paramètres
        ----------
        x_data     : array    — positions mesurées
        v_data     : array    — vitesses mesurées
        x_fine     : array    — positions fines pour tracé
        v_interp   : array    — vitesses interpolées sur x_fine
        w_function : callable — fonction largeur w(x)

        Retourne
        --------
        matplotlib.figure.Figure
        """
        w_fine  = w_function(x_fine)
        vw_fine = v_interp * w_fine

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.scatter(x_data, v_data, color=COLORS["data"], zorder=5, s=60, label="Données v(x)")
        ax1.plot(x_fine, v_interp, color=COLORS["vx"], lw=2, label="v(x) interpolé")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("v (m/s)")
        ax1.set_title("Vitesse v(x)")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(x_fine, vw_fine, color=COLORS["vwx"], lw=2, label="v(x)·w(x)")
        ax2.fill_between(x_fine, vw_fine, alpha=0.25, color=COLORS["vwx"])
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("v·w (m²/s)")
        ax2.set_title("Débit élémentaire v(x)·w(x)")
        ax2.legend()
        ax2.grid(True)

        fig.suptitle("Écoulement dans un canal", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    @staticmethod
    def show():
        """Affiche toutes les figures ouvertes."""
        plt.show()