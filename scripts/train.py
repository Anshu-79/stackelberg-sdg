"""Train / ablate / sweep.
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/ablations/ablation_hyperparams.yaml --override training.beta=0.2
  python scripts/train.py --config configs/ablations/ablation_rho.yaml --sweep
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.experiments import run_one, run_sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", action="append", default=[], help="dot-notation override a.b=c (repeatable)")
    ap.add_argument("--sweep", action="store_true", help="run the full grid in the config's `sweep:` block")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("overrides", nargs="*", help="positional overrides (same as --override)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.override + a.overrides)
    if a.sweep:
        if "sweep" not in cfg:
            sys.exit("config has no `sweep:` block")
        run_sweep(cfg, plots_per_run=not a.no_plots)
    else:
        run_one(cfg, plots=not a.no_plots)


if __name__ == "__main__":
    main()
