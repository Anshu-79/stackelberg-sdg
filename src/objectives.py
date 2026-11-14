"""Pluggable running / terminal functionals."""
from __future__ import annotations
from .registry import OBJECTIVES


@OBJECTIVES.register("icorsi")
class IcorsiObjectives:
    """Regulator: E[int Var(X) + c_alpha alpha^2 dt + c_phi Var(X_T)]  (minimised)
       Bank i  : E[int u - lev/2 (1+alpha) u^2 - c_dev (X^i - Xbar)^2 dt + c_g X^i_T] (maximised)"""
    def __init__(self, c_alpha=0.1, c_phi=2.0, c_dev=0.2, c_g=0.5, lev=1.0):
        self.c_alpha, self.c_phi, self.c_dev, self.c_g, self.lev = c_alpha, c_phi, c_dev, c_g, lev

    def regulator_running(self, var_X, alpha):
        return var_X.mean() + self.c_alpha * (alpha ** 2).mean()

    def regulator_terminal(self, X):
        return self.c_phi * X.var(dim=1).mean()

    def bank_running(self, X, U, alpha, mean_X):
        return U - 0.5 * self.lev * (1.0 + alpha) * U ** 2 - self.c_dev * (X - mean_X) ** 2

    def bank_terminal(self, X):
        return self.c_g * X
