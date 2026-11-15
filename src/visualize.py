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


def plot_exploitability(curve, path, reference=None, title="Nash exploitability", labels=("this run", "DFP baseline")):
    if not curve:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.plot([c["epoch"] for c in curve], [max(c["mean"], 1e-9) for c in curve], "o-", label=labels[0])
    if reference:
        ax.plot([c["epoch"] for c in reference], [max(c["mean"], 1e-9) for c in reference], "s--", color="gray", label=labels[1])
    ax.set(yscale="log", xlabel="epoch", ylabel=r"mean exploitability $\mathcal{E}$", title=title)
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


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


import numpy as np
import torch
from .metrics import collect_paths


def plot_policy_surfaces(solver, path, times=(0.1, 0.5, 0.9), n=60):
    """Contours of alpha(t, Xbar, Var X) at several t, with visited (Xbar, Var) states overlaid."""
    P = collect_paths(solver.dyn, solver.obj, solver.regulator, solver.banks, batch=512)
    m_lo, m_hi = np.quantile(P["mean_paths"], [0.005, 0.995]); v_hi = np.quantile(P["var_paths"], 0.995)
    mg, vg = np.meshgrid(np.linspace(m_lo, m_hi, n), np.linspace(0, v_hi, n))
    mt = torch.tensor(mg.reshape(-1, 1), dtype=torch.float32); vt = torch.tensor(vg.reshape(-1, 1), dtype=torch.float32)
    surf = {}
    with torch.no_grad():
        for t in times:
            surf[t] = solver.regulator(torch.full_like(mt, t), mt, vt).numpy().reshape(n, n)
    lo, hi = min(v.min() for v in surf.values()), max(v.max() for v in surf.values())
    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times) + 0.8, 3.4), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), times):
        k = min(int(round(t / solver.dyn.dt)), solver.dyn.n_steps - 1)
        cs = ax.contourf(mg, vg, surf[t], levels=np.linspace(lo, hi + 1e-9, 15), cmap="viridis")
        ax.scatter(P["mean_paths"][k], P["var_paths"][k], s=2, c="w", alpha=0.35)
        ax.set(title=f"t = {t}", xlabel=r"$\bar X_t$"); ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel(r"$\mathrm{Var}(X_t)$")
    fig.colorbar(cs, ax=axes, label=r"$\alpha(t,\bar X,\mathrm{Var})$")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _corr(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def plot_lead_lag(solver, path, max_lag=3):
    """Dual-axis alpha_t vs Var(X_t) plus cross-correlation of cross-path fluctuations.
    lag > 0 means alpha lags Var(X)."""
    P = collect_paths(solver.dyn, solver.obj, solver.regulator, solver.banks, batch=512)
    t, var, al = P["t"], P["var_paths"], P["alpha_paths"]
    S = len(t)
    dv, da = var - var.mean(1, keepdims=True), al - al.mean(1, keepdims=True)
    lags = list(range(-max_lag, max_lag + 1))
    xc = [(_corr(dv[:S - l], da[l:]) if l >= 0 else _corr(dv[-l:], da[:S + l])) for l in lags]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    j = int(np.argmax(var.max(0)))                                  # path with the largest variance spike
    ax[0].plot(t, var.mean(1), "C0", label=r"$E[\mathrm{Var}(X_t)]$"); ax[0].plot(t, var[:, j], "C0--", alpha=0.5, label="spike path")
    ax[0].set(xlabel="t", ylabel=r"$\mathrm{Var}(X_t)$"); ax[0].tick_params(axis="y", colors="C0")
    a2 = ax[0].twinx(); a2.plot(t, al.mean(1), "C3", label=r"$E[\alpha_t]$"); a2.plot(t, al[:, j], "C3--", alpha=0.5)
    a2.set_ylabel(r"$\alpha_t$"); a2.tick_params(axis="y", colors="C3"); a2.grid(False)
    ax[0].set_title("Regulator response to dispersion"); ax[0].legend(loc="upper left"); a2.legend(loc="upper right")
    ax[1].bar(lags, xc); ax[1].set(xlabel="lag (steps; >0: α lags Var)", ylabel="corr", title="Cross-correlation (cross-path fluctuations)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    best = int(np.nanargmax(np.abs(xc)))
    return {"lags": lags, "xcorr": xc, "peak_lag": lags[best], "peak_corr": xc[best],
            "corr_alpha_var_levels": _corr(var, al)}


def plot_phase_portraits(solver, path, times=(0.1, 0.5, 0.9), max_pts=2000):
    """Gamble-for-resurrection check: relative capital X^i - Xbar vs control u^i.
    A negative slope means under-capitalised banks exert more control effort."""
    P = collect_paths(solver.dyn, solver.obj, solver.regulator, solver.banks, batch=512)
    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 3.4), sharey=True)
    rng = np.random.default_rng(0); slopes = {}
    for ax, t in zip(np.atleast_1d(axes), times):
        k = min(int(round(t / solver.dyn.dt)), solver.dyn.n_steps - 1)
        rel = (P["X_full"][k] - P["mean_paths"][k][:, None]).ravel(); u = P["U_full"][k].ravel()
        sel = rng.choice(len(rel), min(max_pts, len(rel)), replace=False)
        ax.scatter(rel[sel], u[sel], s=3, alpha=0.3)
        edges = np.quantile(rel, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(rel, edges[1:-1]), 0, 9)
        ax.plot([rel[idx == b].mean() for b in range(10)], [u[idx == b].mean() for b in range(10)], "r-o", ms=3)
        slope = float(np.polyfit(rel, u, 1)[0]); slopes[str(t)] = slope
        ax.axvline(0, color="k", lw=0.5); ax.set(title=f"t={t}: slope={slope:.3f}", xlabel=r"$X^i_t-\bar X_t$")
    np.atleast_1d(axes)[0].set_ylabel(r"control $u^i_t$")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return {"slope_u_vs_relative_capital": slopes}


def plot_economic(solver, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    plot_policy_surfaces(solver, out / "policy_surfaces.png")
    return {"lead_lag": plot_lead_lag(solver, out / "lead_lag.png"),
            "phase": plot_phase_portraits(solver, out / "phase_portraits.png")}


def plot_stress(t, base, shock, path, shock_time, rho):
    """base/shock: dicts of series over t (see scripts/run_stress_test.py)."""
    panels = [("var_x", r"cross-sectional $\mathrm{Var}(X_t)$"), ("std_mean", r"std across paths of $\bar X_t$ (systemic risk)"),
              ("q05_mean", r"5% quantile of $\bar X_t$"), ("worst_bank", r"$E[\min_i X^i_t]$"),
              ("alpha", r"$E[\alpha_t]$"), ("mean_x", r"$E[\bar X_t]$")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (k, title) in zip(axes.ravel(), panels):
        ax.plot(t, base[k], label=r"baseline $\rho=0$"); ax.plot(t, shock[k], label=rf"shock $\rho={rho}$")
        ax.axvline(shock_time, color="k", ls=":", lw=0.8); ax.set(title=title, xlabel="t")
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_sweep(results, keys, path, metrics=(("final_exploitability", "final exploitability"),
                                             ("var_XT", r"$\mathrm{Var}(X_T)$"),
                                             ("osc_reg", "regulator-loss oscillation"),
                                             ("osc_expl", "exploitability oscillation"))):
    """results: list of dict(params={key: value}, <summary metrics>)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4))
    if len(keys) == 1:
        xs = [r["params"][keys[0]] for r in results]
        for ax, (m, title) in zip(axes, metrics):
            ax.plot(xs, [r[m] for r in results], "o-"); ax.set(xlabel=keys[0].split(".")[-1], title=title)
    else:
        xv = sorted({r["params"][keys[0]] for r in results}); yv = sorted({r["params"][keys[1]] for r in results})
        for ax, (m, title) in zip(axes, metrics):
            G = np.full((len(yv), len(xv)), np.nan)
            for r in results:
                G[yv.index(r["params"][keys[1]]), xv.index(r["params"][keys[0]])] = r[m]
            im = ax.imshow(G, origin="lower", cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(xv)), [str(v) for v in xv]); ax.set_yticks(range(len(yv)), [str(v) for v in yv])
            ax.set(xlabel=keys[0].split(".")[-1], ylabel=keys[1].split(".")[-1], title=title); ax.grid(False)
            for i in range(len(yv)):
                for j in range(len(xv)):
                    ax.text(j, i, f"{G[i, j]:.3g}", ha="center", va="center", color="w", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_all(solver, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if solver.history:
        plot_training(solver.history, out / "training.png")
    plot_exploitability(solver.expl_curve, out / "exploitability.png")
    if solver.cfg.training.get("mode", "dfp") == "coopt":
        ref = None
        rp = Path(solver.cfg.training.get("reference_dir", "outputs/baseline")) / "exploitability.json"
        if rp.exists():
            import json
            ref = json.loads(rp.read_text())
        plot_exploitability(solver.expl_curve, out / "exploitability_failure.png", reference=ref,
                            title="Naive co-optimisation: no Nash equilibrium", labels=("co-opt", "DFP baseline"))
    d = solver.diagnostics()
    plot_trajectories(d, out / "trajectories.png")
    plot_symmetry(d, out / "symmetry.png")
    d["economic"] = plot_economic(solver, out)
    return d
