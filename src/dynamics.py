"""SDE dynamics, Euler-Maruyama integrator and the differentiable game rollout.

dX^i = [kappa (Xbar - X^i) + u^i - alpha] dt + sigma (rho dW^0 + sqrt(1-rho^2) dW^i)
"""
from __future__ import annotations
import math
import torch
from .registry import DYNAMICS


@DYNAMICS.register("interbank_ou")
class InterbankOU:
    def __init__(self, n_banks=10, T=1.0, n_steps=20, kappa=0.5, sigma=0.2,
                 rho=0.0, x0_std=0.5, shock_time=None, shock_rho=0.0, device="cpu"):
        assert n_banks >= 2, "Var(X_t) across banks requires n_banks >= 2"
        assert 0.0 <= rho <= 1.0 and 0.0 <= shock_rho <= 1.0
        self.shock_time, self.shock_rho = shock_time, shock_rho   # optional contagion regime switch
        self.n_banks, self.T, self.n_steps = n_banks, T, n_steps
        self.kappa, self.sigma, self.rho, self.x0_std = kappa, sigma, rho, x0_std
        self.dt = T / n_steps
        self.device = torch.device(device)

    def initial_state(self, batch, generator=None):
        return torch.randn(batch, self.n_banks, generator=generator, device=self.device) * self.x0_std

    def noise(self, batch, generator=None):
        """Returns (dW0 [B,1], dWi [B,N]); dW0 is zero (and not sampled) when rho == 0
        so the RNG stream matches the original notebook."""
        s = math.sqrt(self.dt)
        dWi = torch.randn(batch, self.n_banks, generator=generator, device=self.device) * s
        if max(self.rho, self.shock_rho if self.shock_time is not None else 0.0) > 0:
            dW0 = torch.randn(batch, 1, generator=generator, device=self.device) * s
        else:
            dW0 = torch.zeros(batch, 1, device=self.device)
        return dW0, dWi

    def drift(self, t, X, mean_X, U, alpha):
        return self.kappa * (mean_X - X) + U - alpha

    def diffusion(self, t, X):
        return self.sigma

    def rho_at(self, t=None):
        if self.shock_time is not None and t is not None and t >= self.shock_time - 1e-9:
            return self.shock_rho
        return self.rho

    def step(self, X, drift, dW0, dWi, t=None):
        rho = self.rho_at(t)
        dW = rho * dW0 + math.sqrt(1.0 - rho ** 2) * dWi
        return X + drift * self.dt + self.diffusion(None, X) * dW


def simulate(dyn, obj, regulator, bank_fn, batch, *, y0=None, leader_grad=False,
             detach_mean=True, generator=None, record=False):
    """Roll the game from 0 to T.

    bank_fn(t, X, mean_X, alpha) -> (U, Z, Z0) each [B,N] (Z, Z0 may be None).
    regulator(t, mean_X, var_X)  -> alpha [B,1].
    y0: optional [N] initial BSDE value; if given and Z is not None, the forward
        adjoint process Y is propagated and a terminal residual is returned.
    Returns dict(reg_loss, payoffs[N], bsde_residual[N]|None, X_T, rec).
    """
    N, dt = dyn.n_banks, dyn.dt
    X = dyn.initial_state(batch, generator)
    reg_run = X.new_zeros(())
    rewards = X.new_zeros(batch, N)
    Y = y0.view(1, N).expand(batch, N) if y0 is not None else None
    rec = {k: [] for k in ("mean_X", "var_X", "alpha", "U", "drift_mean", "drift_std",
                                   "X_full", "U_full", "alpha_paths", "mean_paths", "var_paths")} if record else None
    used_bsde = False

    for k in range(dyn.n_steps):
        t = torch.full((batch, 1), k * dt, device=dyn.device)
        mean_X = X.mean(dim=1, keepdim=True)
        var_X = X.var(dim=1, keepdim=True)
        alpha = regulator(t, mean_X, var_X)
        if not leader_grad:
            alpha = alpha.detach()
        m_in = mean_X.detach() if detach_mean else mean_X

        U, Z, Z0 = bank_fn(t, X, m_in, alpha)

        reg_run = reg_run + obj.regulator_running(var_X, alpha) * dt
        f = obj.bank_running(X, U, alpha, m_in)
        rewards = rewards + f * dt

        dW0, dWi = dyn.noise(batch, generator)
        drift = dyn.drift(t, X, m_in, U, alpha)
        if Y is not None and Z is not None:
            Y = Y - f * dt + Z * dWi + Z0 * dW0
            used_bsde = True
        if record:
            rec["mean_X"].append(mean_X.mean().detach()); rec["var_X"].append(var_X.mean().detach())
            rec["alpha"].append(alpha.mean().detach()); rec["U"].append(U.mean(0).detach())
            rec["drift_mean"].append(drift.mean().detach()); rec["drift_std"].append(drift.std().detach())
            rec["X_full"].append(X.detach()); rec["U_full"].append(U.detach())
            rec["alpha_paths"].append(alpha.detach().squeeze(-1))
            rec["mean_paths"].append(mean_X.detach().squeeze(-1)); rec["var_paths"].append(var_X.detach().squeeze(-1))
        X = dyn.step(X, drift, dW0, dWi, t=k * dt)

    g = obj.bank_terminal(X)
    return {
        "reg_loss": reg_run + obj.regulator_terminal(X),
        "payoffs": (rewards + g).mean(dim=0),
        "bsde_residual": ((Y - g) ** 2).mean(0) if used_bsde else None,
        "X_T": X.detach(),
        "rec": {k: torch.stack(v) for k, v in rec.items()} if record else None,
    }
