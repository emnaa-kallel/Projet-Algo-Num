class AdaptiveIntegration:

    def __init__(self, tol=1e-6, max_depth=20):
        self.tol = tol
        self.max_depth = max_depth

    def _simpson_rule(self, f, a, b):
        c = (a + b) / 2
        return (b - a) / 6 * (f(a) + 4*f(c) + f(b))

    def adaptive_simpson(self, f, a, b, tol=None, depth=0):
        if tol is None:
            tol = self.tol

        c = (a + b) / 2

        S = self._simpson_rule(f, a, b)
        S_left = self._simpson_rule(f, a, c)
        S_right = self._simpson_rule(f, c, b)

        if depth >= self.max_depth or abs(S_left + S_right - S) < 15 * tol:
            return S_left + S_right + (S_left + S_right - S)/15

        return (self.adaptive_simpson(f, a, c, tol/2, depth+1) +
                self.adaptive_simpson(f, c, b, tol/2, depth+1))