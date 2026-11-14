"""Publication-style diagnostic plots."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def plot_training(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ep, [h["reg_loss"] for h in history]); ax[0].set(title="Regulator loss $J_{reg}$", xlabel="epoch")
    ax[1].plot(ep, [h["avg_bank_payoff"] for h in history], color="C1"); ax[1].set(title="Avg bank payoff", xlabel="epoch")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_exploitability(curve, path):
    if not curve:
        return
    ep = [c["epoch"] for c in curve]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(ep, [max(c["mean"], 1e-12) for c in curve], "o-")
    ax.set(yscale="log", xlabel="epoch", ylabel=r"mean exploitability $\mathcal{E}$", title="Nash exploitability")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_trajectories(diag, path):
    t = diag["t"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(t, diag["alpha"]); ax[0].set(title=r"$\bar\alpha_t$", xlabel="t")
    ax[1].plot(t, diag["mean_X"], color="C1"); ax[1].set(title=r"$E[\bar X_t]$", xlabel="t")
    ax[2].plot(t, diag["var_X"], color="C2"); ax[2].set(title=r"$E[\mathrm{Var}(X_t)]$", xlabel="t")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_symmetry(diag, path):
    n = len(diag["U_bank"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(range(n), diag["U_bank"]); ax[0].set(title=f"time-avg control (std={diag['symmetry_dispersion_u']:.2e})", xlabel="bank")
    ax[1].bar(range(n), diag["payoff_bank"], color="C1"); ax[1].set(title=f"payoff (std={diag['symmetry_dispersion_payoff']:.2e})", xlabel="bank")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_all(solver, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if solver.history:
        plot_training(solver.history, out / "training.png")
    plot_exploitability(solver.expl_curve, out / "exploitability.png")
    d = solver.diagnostics()
    plot_trajectories(d, out / "trajectories.png")
    plot_symmetry(d, out / "symmetry.png")
    return d
