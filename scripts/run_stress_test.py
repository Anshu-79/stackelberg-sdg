"""Systemic contagion stress test: rho jumps from 0 to --rho at t = shock_time (policies trained at rho=0).
  python scripts/run_stress_test.py --rho 0.8"""
import argparse, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import load_config
from src.experiments import set_seed
from src.solver import StackelbergSolver
from src.metrics import collect_paths
from src.visualize import plot_stress


def series(P):
    X, m = P["X_full"], P["mean_paths"]
    return {"var_x": P["var_paths"].mean(1), "std_mean": m.std(1), "q05_mean": np.quantile(m, 0.05, axis=1),
            "worst_bank": X.min(2).mean(1), "alpha": P["alpha_paths"].mean(1), "mean_x": m.mean(1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/systemic_shock.yaml")
    ap.add_argument("--rho", type=float, default=None, help="post-shock correlation (default: stress.shock_rho)")
    ap.add_argument("--shock-time", type=float, default=None)
    ap.add_argument("--ckpt", default=None, help="use a trained checkpoint instead of training from the config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.override)
    st = cfg.get("stress", {})
    rho = a.rho if a.rho is not None else st.get("shock_rho", 0.8)
    t_shock = a.shock_time if a.shock_time is not None else st.get("shock_time", 0.5 * cfg.game.T)
    n_paths, seed = st.get("n_paths", 4096), st.get("seed", 11)
    out = Path(a.out or cfg.training.out_dir); out.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.training.seed)
    if a.ckpt:
        solver = StackelbergSolver.from_checkpoint(a.ckpt)
    else:
        print(f"[stress] training baseline policy (rho={cfg.game.rho}) for {cfg.training.epochs} epochs ...")
        solver = StackelbergSolver(cfg); solver.train(); solver.save(out)
    base_dyn = solver.dyn
    shock_dyn = copy.copy(base_dyn); shock_dyn.shock_time, shock_dyn.shock_rho = t_shock, rho
    print(f"[stress] simulating {n_paths} paths with common random numbers; rho: {base_dyn.rho} -> {rho} at t={t_shock}")
    Pb = collect_paths(base_dyn, solver.obj, solver.regulator, solver.banks, batch=n_paths, seed=seed)
    Ps = collect_paths(shock_dyn, solver.obj, solver.regulator, solver.banks, batch=n_paths, seed=seed)
    sb, ss = series(Pb), series(Ps)
    plot_stress(Pb["t"], sb, ss, out / "stress_trajectories.png", t_shock, rho)
    summ = {"rho_shock": rho, "shock_time": t_shock, "n_paths": n_paths}
    for k in sb:
        summ[k] = {"baseline_T": float(sb[k][-1]), "shock_T": float(ss[k][-1]),
                   "ratio_shock_over_baseline": float(ss[k][-1] / sb[k][-1]) if sb[k][-1] != 0 else None}
    (out / "stress_summary.json").write_text(json.dumps(summ, indent=2))
    for k in ("var_x", "std_mean", "q05_mean", "worst_bank", "alpha"):
        print(f"  {k:<11} baseline={summ[k]['baseline_T']:+.4f}  shock={summ[k]['shock_T']:+.4f}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
