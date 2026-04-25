import numpy as np

class GaussQuadrature:

    @staticmethod
    def gauss_legendre_2(f, a, b):
        x1 = -1/np.sqrt(3)
        x2 = 1/np.sqrt(3)

        t1 = (b-a)/2 * x1 + (a+b)/2
        t2 = (b-a)/2 * x2 + (a+b)/2

        return (b-a)/2 * (f(t1) + f(t2))

    @staticmethod
    def gauss_legendre_3(f, a, b):
        """
        Quadrature de Gauss-Legendre à 3 points.
        
        Points et poids pour n=3:
        - x1 = -√(3/5) ≈ -0.7745966692, w1 = 5/9
        - x2 = 0, w2 = 8/9
        - x3 = +√(3/5) ≈ 0.7745966692, w3 = 5/9
        
        Paramètres
        ----------
        f : callable
            Fonction à intégrer.
        a, b : float
            Bornes d'intégration.
            
        Retourne
        -------
        float
            Approximation de l'intégrale.
        """
        sqrt_35 = np.sqrt(3/5)
        
        x1 = -sqrt_35
        x2 = 0.0
        x3 = sqrt_35
        
        w1 = 5/9
        w2 = 8/9
        w3 = 5/9
        
        # Transformation vers [a, b]
        t1 = (b-a)/2 * x1 + (a+b)/2
        t2 = (b-a)/2 * x2 + (a+b)/2
        t3 = (b-a)/2 * x3 + (a+b)/2
        
        return (b-a)/2 * (w1*f(t1) + w2*f(t2) + w3*f(t3))