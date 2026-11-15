"""Scalability profiling and Monte Carlo convergence validation."""
from __future__ import annotations
import json, random, resource, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Cfg, apply_overrides
from .solver import StackelbergSolver
from .dynamics import simulate


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def profile_worker(cfg_dict, epochs):
    """Runs in a fresh process so that ru_maxrss is a clean per-configuration peak."""
    cfg = Cfg(cfg_dict)
    random.seed(0); torch.manual_seed(0)
    cuda = cfg.training.device.startswith("cuda")
    base = _rss_mb()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    solver = StackelbergSolver(cfg)
    solver.train_epoch()                                  # warm-up
    t0 = time.perf_counter()
    for _ in range(epochs):
        solver.train_epoch()
    if cuda:
        torch.cuda.synchronize()
    return {"time_per_epoch": (time.perf_counter() - t0) / epochs,
            "n_params": sum(p.numel() for p in solver.banks.parameters()),
            "peak_rss_mb": _rss_mb(), "rss_base_mb": base,
            "peak_cuda_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None}


def run_n_scaling(cfg, out_dir, isolate=True, log=print):
    sc = cfg.scaling
    res = []
    for mode in sc.modes:
        for N in sc.n_values:
            d = apply_overrides(cfg.to_dict(), [f"game.n_banks={N}", f"banks.mode={mode}"])
            log(f"[n_scaling] mode={mode:<11} N={N:<4} ...")
            if isolate:
                with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
                    r = pool.apply(profile_worker, (d, sc.bench_epochs))
            else:
                r = profile_worker(d, sc.bench_epochs)
            r.update(mode=mode, N=N); res.append(r)
            log(f"    {r['time_per_epoch']:.3f} s/epoch | params {r['n_params']} | peak RSS {r['peak_rss_mb']:.0f} MB")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "n_scaling_profile.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for mode in sc.modes:
        rr = [r for r in res if r["mode"] == mode]; N = [r["N"] for r in rr]
        ax[0].plot(N, [r["time_per_epoch"] for r in rr], "o-", label=mode)
        ax[1].plot(N, [r["peak_cuda_mb"] if r["peak_cuda_mb"] is not None else r["peak_rss_mb"] - r["rss_base_mb"] for r in rr], "o-", label=mode)
        ax[2].plot(N, [r["n_params"] for r in rr], "o-", label=mode)
    ax[0].set(xlabel="N banks", ylabel="s / epoch", title="Wall-clock per epoch", yscale="log")
    ax[1].set(xlabel="N banks", ylabel="MB", title="Peak memory (CUDA, else RSS growth over baseline)")
    ax[2].set(xlabel="N banks", title="Bank parameters", yscale="log"); ax[0].legend()
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "n_scaling_profile.png", dpi=150); plt.close(fig)
    return res


def mc_chunks(solver, B, seed, chunk):
    """Equal-sized chunk estimates of (Y0 := avg bank value, E[mean_i X_T])."""
    dyn, fn = solver.dyn, (lambda t, X, m, a: solver.banks(t, X, m, a))
    n = max(1, B // chunk); b = B // n
    ys, xs = [], []
    for j in range(n):
        gen = torch.Generator(device=dyn.device).manual_seed(seed + j)
        with torch.no_grad():
            o = simulate(dyn, solver.obj, solver.regulator, fn, b, generator=gen)
        ys.append(o["payoffs"].mean().item()); xs.append(o["X_T"].mean().item())
    return np.array(ys), np.array(xs)


def run_mc_rate(cfg, out_dir, ckpt=None, log=print):
    mc = cfg.mc
    if ckpt:
        solver = StackelbergSolver.from_checkpoint(ckpt)
    else:
        cfg.training.epochs, cfg.training.eval_every = mc.train_epochs, 0
        random.seed(mc.seed); torch.manual_seed(mc.seed)
        solver = StackelbergSolver(cfg); log(f"[mc_rate] training {mc.train_epochs} epochs ..."); solver.train(log)
    log(f"[mc_rate] ground truth with B_ref={mc.ref_batch} ...")
    ry, rx = mc_chunks(solver, mc.ref_batch, 10 ** 6, mc.chunk)
    truth = {"Y0": ry.mean(), "XbarT": rx.mean()}
    truth_se = {"Y0": ry.std(ddof=1) / np.sqrt(len(ry)) if len(ry) > 1 else float("nan"),
                "XbarT": rx.std(ddof=1) / np.sqrt(len(rx)) if len(rx) > 1 else float("nan")}
    rmse = {"Y0": [], "XbarT": []}
    for B in mc.batches:
        ey, ex = [], []
        for r in range(mc.reps):
            y, x = mc_chunks(solver, B, 1000 + 7919 * r + B, B)
            ey.append(y[0] - truth["Y0"]); ex.append(x[0] - truth["XbarT"])
        rmse["Y0"].append(float(np.sqrt(np.mean(np.square(ey))))); rmse["XbarT"].append(float(np.sqrt(np.mean(np.square(ex)))))
        log(f"    B={B:<5} RMSE Y0={rmse['Y0'][-1]:.3e}  RMSE XbarT={rmse['XbarT'][-1]:.3e}")
    slopes = {k: float(np.polyfit(np.log(mc.batches), np.log(v), 1)[0]) for k, v in rmse.items()}
    log(f"[mc_rate] fitted log-log slopes (theory -0.5): {slopes}")
    res = {"batches": list(mc.batches), "rmse": rmse, "slopes": slopes, "truth": {k: float(v) for k, v in truth.items()},
           "truth_se": {k: float(v) for k, v in truth_se.items()}, "reps": mc.reps}
    if solver.banks.bsde:
        y0 = float(solver.banks.y0_vector().mean())
        res["learned_Y0_param"] = {"value": y0, "abs_error_vs_truth": abs(y0 - float(truth["Y0"]))}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "mc_convergence.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(figsize=(5, 3.8)); B = np.array(mc.batches, float)
    for k, mk in (("Y0", "o"), ("XbarT", "s")):
        ax.loglog(B, rmse[k], mk + "-", label=f"{k} (slope {slopes[k]:.2f})")
    ax.loglog(B, rmse["Y0"][0] * (B / B[0]) ** -0.5, "k--", label=r"$\mathcal{O}(B^{-1/2})$")
    ax.set(xlabel="batch size B", ylabel="RMSE vs. high-res truth", title="Monte Carlo convergence"); ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "mc_convergence_log-log.png", dpi=150); plt.close(fig)
    return res
