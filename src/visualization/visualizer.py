"""
src/visualization.py
====================
Classe unique de visualisation — remplace visualizer.py et visualize_problems.py.

Toutes les méthodes de tracé sont ici. Les fonctions standalone de
visualize_problems.py sont intégrées dans main.py (generate_visualizations).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec   # conservé pour extensions futures

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

# Palette nommée (usage interne)
_COLORS = {
    "lagrange":  "#2196F3",
    "newton":    "#FF5722",
    "data":      "#4CAF50",
    "model":     "#9C27B0",
    "uniform":   "#F44336",
    "chebyshev": "#00BCD4",
    "vx":        "#00BCD4",
    "vwx":       "#FF9800",
}

_METHOD_STYLES = [
    ("#E63946", "-",  "o"),
    ("#2196F3", "--", "s"),
    ("#4CAF50", "-",  "^"),
    ("#FF9800", ":",  "D"),
    ("#9C27B0", "-.", "v"),
    ("#00BCD4", "--", "P"),
]

# Palette indexée pour méthodes multiples
_PALETTE = ["#E63946", "#2196F3", "#4CAF50", "#FF9800"]

def _get_method_style(idx):
    s = _METHOD_STYLES[idx % len(_METHOD_STYLES)]
    return {"color": s[0], "linestyle": s[1], "marker": s[2]}
 
 
def _estimate_slope(n_values, errors):
    """Régression linéaire en log-log → taux de convergence empirique."""
    n_arr = np.array(n_values, dtype=float)
    e_arr = np.array(errors,   dtype=float)
    mask  = e_arr > 1e-15
    if mask.sum() < 2:
        return None
    return -np.polyfit(np.log2(n_arr[mask]), np.log2(e_arr[mask]), 1)[0]
 

class Visualizer:
    """
    Visualiseur central du projet d'analyse numérique.

    Paramètres
    ----------
    style   : str   — style matplotlib (défaut 'seaborn-v0_8-darkgrid')
    figsize : tuple — taille par défaut des figures (défaut (10, 6))
    """

    def __init__(self, style="seaborn-v0_8-darkgrid", figsize=(10, 6)):
        self.figsize = figsize
        self.style = style
        self.results_dir = "results"
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

    # ─── 1. Comparaison interpolations ────────────────────────────────────────

    def plot_interpolation_comparison(self, x_data, y_data, interpolators, x_fine, title):
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.scatter(x_data, y_data, color=_COLORS["data"], zorder=6, s=70,
                        label="Données mesurées")
                color_map = {"lagrange": _COLORS["lagrange"], "newton": _COLORS["newton"]}
                for idx, (name, y_interp) in enumerate(interpolators.items()):
                    color = color_map.get(name.lower(), _METHOD_STYLES[idx % len(_METHOD_STYLES)][0])
                    ax.plot(x_fine, y_interp, lw=2, label=name, color=color)
                ax.set_title(title, fontsize=13, fontweight="bold")
                ax.set_xlabel("x"); ax.set_ylabel("y")
                ax.legend(fontsize=10); ax.grid(True, linestyle="--", alpha=0.4)
                fig.tight_layout()
            return fig

    # ─── 2. Comparaison méthodes d'intégration ────────────────────────────────

    def plot_integration_comparison(
            self,
            f,
            a,
            b,
            n_values,
            methods_dict,
            exact_value,
            title="Comparaison des méthodes d'intégration",
        ):
            """
            Calcule les erreurs pour chaque méthode, puis délègue à plot_convergence.
    
            Interface différente de plot_convergence :
            → reçoit les callables bruts + valeur exacte
            → calcule lui-même errors_dict
            → appelle self.plot_convergence(errors_dict, ...)
    
            Paramètres
            ----------
            f            : callable — fonction à intégrer
            a, b         : float    — bornes
            n_values     : list     — valeurs de n
            methods_dict : dict     — {"nom": callable(f, a, b, n)}
            exact_value  : float    — valeur exacte de l'intégrale
            title        : str
    
            Retourne
            --------
            matplotlib.figure.Figure
            """
            errors_dict = {
                name: [abs(method(f, a, b, n) - exact_value) for n in n_values]
                for name, method in methods_dict.items()
            }
            return self.plot_convergence(
                n_values, errors_dict,
                title=title,
                xlabel="n (subdivisions)",
                ylabel="Erreur absolue",
            )

    # ─── 3. Phénomène de Runge ────────────────────────────────────────────────

    def plot_runge_phenomenon(self, x_fine, y_true, interpolations):
        """
        Visualise le phénomène de Runge : fonction exacte vs interpolations.

        Paramètres
        ----------
        x_fine         : array-like — domaine continu
        y_true         : array-like — valeurs exactes
        interpolations : dict       — {nom: valeurs interpolées}
                         (les clés 'x_fine' et 'y_true' sont ignorées)
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            ax.plot(x_fine, y_true, "k--", lw=2, label="Fonction exacte")

            for idx, (name, y_interp) in enumerate(interpolations.items()):
                if name in ("x_fine", "y_true"):
                    continue
                color = _PALETTE[idx % len(_PALETTE)]
                ax.plot(x_fine, y_interp, lw=2, label=name, color=color)

            ax.set_title("Phénomène de Runge", fontsize=13, fontweight="bold")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()

        # Sauvegarde automatique dans results/
        os.makedirs(self.results_dir, exist_ok=True)
        filepath = os.path.join(self.results_dir, "07_runge_phenomenon.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ─── 4. Convergence ───────────────────────────────────────────────────────

    def plot_convergence(
        self,
        n_values,
        errors_dict,
        title="Étude de convergence — Erreur en fonction de n",
        xlabel="Nombre de subdivisions n",
        ylabel="Erreur absolue",
        show_theory=True,
        show_slopes=True,
    ):
        """
        Courbes de convergence en log-log avec :
          — styles distincts par méthode (couleur + tiret + marqueur)
          — pentes théoriques de référence O(hⁿ) en gris pointillé
          — annotation automatique du taux empirique estimé sur chaque courbe
 
        Paramètres
        ----------
        n_values    : list  — valeurs de n
        errors_dict : dict  — {"nom_méthode": [erreurs pour chaque n]}
        title       : str
        xlabel      : str
        ylabel      : str
        show_theory : bool  — tracer les pentes théoriques (défaut True)
        show_slopes : bool  — annoter les taux empiriques (défaut True)
 
        Retourne
        --------
        matplotlib.figure.Figure
        """
        n_arr = np.array(n_values, dtype=float)
 
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
 
            # ── Pentes théoriques (gris discret, ne polluent pas la légende) ──
            if show_theory:
                first_errs = next(iter(errors_dict.values()))
                e_anchor = max(first_errs[0], 1e-15)
                for order, lbl in [(1, "O(h¹)"), (2, "O(h²)"), (4, "O(h⁴)")]:
                    ref = e_anchor * (n_arr[0] / n_arr) ** order
                    ax.loglog(n_arr, ref, linestyle=":", lw=1,
                              color="#444c56", alpha=0.65, label=lbl)
 
            # ── Courbes des méthodes ──────────────────────────────────────────
            for idx, (name, errors) in enumerate(errors_dict.items()):
                e_arr = np.where(np.array(errors, dtype=float) < 1e-15,
                                 1e-15, np.array(errors, dtype=float))
                sty = _get_method_style(idx)
                ax.loglog(n_arr, e_arr, linewidth=2.2, markersize=7,
                          label=name, **sty)
 
                if show_slopes:
                    slope = _estimate_slope(n_values, errors)
                    if slope is not None:
                        ax.annotate(
                            f"  ≈ O(h{slope:.1f})",
                            xy=(n_arr[-1], e_arr[-1]),
                            fontsize=8, color=sty["color"], va="center",
                        )
 
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(fontsize=9, ncol=2)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.tick_params(axis="both", which="major", labelsize=10)
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
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)

            ax.scatter(t_data, T_data,
                       color=_COLORS["data"], zorder=6, s=70, label="Données mesurées")
            ax.plot(t_fine, T_interp,
                    color=_COLORS["newton"], lw=2, label="Interpolation Newton")
            ax.plot(t_fine, T_model,
                    color=_COLORS["model"], lw=2, linestyle="--",
                    label=f"Modèle exp (k={k_opt:.4f})")
            ax.axhline(20.0, color="#555", linestyle="-.", lw=1, label="T_amb = 20°C")

            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Température (°C)")
            ax.set_title("Refroidissement du composant électronique",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()

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

        with plt.style.context(self.style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

            ax1.scatter(x_data, v_data,
                        color=_COLORS["data"], zorder=5, s=60, label="Données v(x)")
            ax1.plot(x_fine, v_interp,
                     color=_COLORS["vx"], lw=2, label="v(x) interpolé")
            ax1.set_xlabel("x (m)")
            ax1.set_ylabel("v (m/s)")
            ax1.set_title("Vitesse v(x)")
            ax1.legend(fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.4)

            ax2.plot(x_fine, vw_fine,
                     color=_COLORS["vwx"], lw=2, label="v(x)·w(x)")
            ax2.fill_between(x_fine, vw_fine, alpha=0.25, color=_COLORS["vwx"])
            ax2.set_xlabel("x (m)")
            ax2.set_ylabel("v·w (m²/s)")
            ax2.set_title("Débit élémentaire v(x)·w(x)")
            ax2.legend(fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.4)

            fig.suptitle("Écoulement dans un canal", fontsize=13, fontweight="bold")
            fig.tight_layout()

        return fig

    # ─── Utilitaire ───────────────────────────────────────────────────────────

    @staticmethod
    def show():
        """Affiche toutes les figures ouvertes."""
        plt.show()