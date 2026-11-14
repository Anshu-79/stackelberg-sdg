"""python scripts/train.py --config configs/default.yaml [a.b=c ...]"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.config import load_config
from src.solver import StackelbergSolver
from src.visualize import plot_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dot-notation overrides, e.g. banks.mode=shared game.n_banks=50")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    s = cfg.training.seed
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    solver = StackelbergSolver(cfg)
    solver.train()
    solver.save(cfg.training.out_dir)
    if not a.no_plots:
        d = plot_all(solver, cfg.training.out_dir)
        (Path(cfg.training.out_dir) / "diagnostics.json").write_text(json.dumps(d))
    print("saved to", cfg.training.out_dir)


if __name__ == "__main__":
    main()
