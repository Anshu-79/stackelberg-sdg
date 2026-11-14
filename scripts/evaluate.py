"""python scripts/evaluate.py --ckpt runs/default/checkpoint.pt [--k-br 200 --fresh] [overrides]"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.solver import StackelbergSolver
from src.visualize import plot_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k-br", type=int, default=None)
    ap.add_argument("--n-banks", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="best-response net from fresh init")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    solver = StackelbergSolver.from_checkpoint(a.ckpt, a.overrides)
    seed = solver.cfg.training.seed
    random.seed(seed); torch.manual_seed(seed)
    e = solver.evaluate_exploitability(fresh=a.fresh or None, k_br=a.k_br, n_banks=a.n_banks)
    out = Path(a.out or Path(a.ckpt).parent / "eval"); out.mkdir(parents=True, exist_ok=True)
    d = plot_all(solver, out)
    (out / "exploitability_eval.json").write_text(json.dumps(e))
    (out / "diagnostics.json").write_text(json.dumps(d))
    print(f"mean exploitability = {e['mean']:.6f}  per-bank = {np.round(e['per_bank'], 6).tolist()}")
    print(f"symmetry dispersion (u) = {d['symmetry_dispersion_u']:.3e}; saved to {out}")


if __name__ == "__main__":
    main()
