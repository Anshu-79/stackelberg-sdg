"""Shared experiment runner: single runs, grid sweeps, summary metrics."""
from __future__ import annotations
import itertools, json, random, re, time
from pathlib import Path
import numpy as np
import torch

from .config import Cfg, apply_overrides
from .solver import StackelbergSolver
from . import visualize as viz


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _get(cfg, key):
    node = cfg
    for p in key.split("."):
        node = node[p]
    return node


def resolve_path(template, cfg):
    """'outputs/x_{fictitious_play.beta}' -> value substitution from cfg."""
    return re.sub(r"\{([\w.]+)\}", lambda m: str(_get(cfg, m.group(1))), template)


def summarize(solver, diag, seconds):
    reg = np.array([h["reg_loss"] for h in solver.history])
    half = reg[len(reg) // 2:]
    ex = np.array([c["mean"] for c in solver.expl_curve]) if solver.expl_curve else np.array([])
    exh = ex[len(ex) // 2:] if len(ex) >= 4 else ex
    return {
        "final_exploitability": float(ex[-1]) if len(ex) else float("nan"),
        "final_reg_loss": float(reg[-10:].mean()),
        "var_XT": diag["var_XT"], "mean_XT": diag["mean_XT"],
        "osc_reg": float(np.std(np.diff(half))) if len(half) > 2 else float("nan"),
        "osc_expl": float(np.mean(np.abs(np.diff(exh)))) if len(exh) > 1 else float("nan"),
        "train_seconds": seconds,
    }


def run_one(cfg, plots=True, log=print):
    out = Path(resolve_path(cfg.training.out_dir, cfg)); cfg.training.out_dir = str(out)
    set_seed(cfg.training.seed)
    log(f"[run] -> {out}  (banks={cfg.banks.mode}, N={cfg.game.n_banks}, rho={cfg.game.rho}, "
        f"mode={cfg.training.get('mode', 'dfp')}, beta={cfg.fictitious_play.beta}, inner={cfg.fictitious_play.inner_steps_per_epoch})")
    solver = StackelbergSolver(cfg)
    t0 = time.time(); solver.train(log); secs = time.time() - t0
    solver.save(out)
    diag = viz.plot_all(solver, out) if plots else solver.diagnostics()
    summ = summarize(solver, diag, secs)
    if cfg.training.get("mode", "dfp") == "coopt" and cfg.training.get("settle_epochs", 0) > 0:
        before = solver.eval_reg_loss()
        log(f"[coopt] frozen regulator; followers best-respond selfishly for {cfg.training.settle_epochs} epochs ...")
        solver.settle_followers(cfg.training.settle_epochs)
        after = solver.eval_reg_loss()
        summ.update(reg_loss_coopt=before, reg_loss_after_followers_settle=after, stackelberg_gap=after - before)
        log(f"[coopt] J_reg: {before:.4f} (co-opt) -> {after:.4f} once followers act selfishly")
    (out / "diagnostics.json").write_text(json.dumps(diag))
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    log(f"[run] summary: {json.dumps(summ)}")
    return summ, solver


def run_sweep(cfg, log=print, plots_per_run=False):
    sweep = cfg.sweep.to_dict() if hasattr(cfg.sweep, "to_dict") else dict(cfg.sweep)
    keys = list(sweep)
    root = Path(resolve_path(cfg.training.out_dir, cfg)).parent if cfg.get("sweep_out_dir") else Path("outputs/sweep")
    results = []
    for combo in itertools.product(*[sweep[k] for k in keys]):
        ov = [f"{k}={v}" for k, v in zip(keys, combo)]
        d = apply_overrides(cfg.to_dict(), ov); d.pop("sweep", None)
        c = Cfg(d)
        c.training.out_dir = cfg.get("sweep_out_dir", cfg.training.out_dir)
        summ, _ = run_one(c, plots=plots_per_run, log=log)
        results.append({"params": dict(zip(keys, combo)), **summ})
    out = Path(resolve_path(cfg.sweep_out_dir, apply_overrides_cfg(cfg, keys, [sweep[k][0] for k in keys]))).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_summary.json").write_text(json.dumps(results, indent=2))
    viz.plot_sweep(results, keys, out / "sweep_summary.png")
    log(f"[sweep] wrote {out}/sweep_summary.json|png")
    return results


def apply_overrides_cfg(cfg, keys, vals):
    return Cfg(apply_overrides(cfg.to_dict(), [f"{k}={v}" for k, v in zip(keys, vals)]))
