import numpy as np

class NewtonCotes:

    @staticmethod
    def rectangle(f, a, b, n=1):
        h = (b - a) / n
        s = 0
        for i in range(n):
            s += f(a + i*h)
        return h * s

    @staticmethod
    def trapezoidal(f, a, b, n=1):
        h = (b - a) / n
        s = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            s += f(a + i*h)
        return h * s

    @staticmethod
    def simpson(f, a, b, n=2):
        if n % 2 != 0:
            raise ValueError("n doit être pair")

        h = (b - a) / n
        s = f(a) + f(b)

        for i in range(1, n):
            if i % 2 == 0:
                s += 2 * f(a + i*h)
            else:
                s += 4 * f(a + i*h)

        return (h / 3) * s

    @staticmethod
    def simpson_38(f, a, b, n=3):
        """
        Méthode de Simpson 3/8 (formule composite).
        
        La formule de Simpson 3/8 utilise 4 points par intervalle de 3 sous-intervalles.
        Formule : ∫f(x)dx ≈ (3h/8)[f(x₀) + 3f(x₁) + 3f(x₂) + f(x₃)]
        
        Paramètres
        ----------
        f : callable
            Fonction à intégrer.
        a, b : float
            Bornes d'intégration.
        n : int
            Nombre de sous-intervalles (doit être multiple de 3).
            
        Retourne
        -------
        float
            Approximation de l'intégrale.
        """
        if n % 3 != 0:
            raise ValueError("n doit être un multiple de 3 pour Simpson 3/8")
        
        h = (b - a) / n
        s = f(a) + f(b)
        
        for i in range(1, n):
            if i % 3 == 0:
                s += 2 * f(a + i*h)
            else:
                s += 3 * f(a + i*h)
        
        return (3 * h / 8) * s