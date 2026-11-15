"""Economic-insight figures from a trained checkpoint (policy surfaces, lead-lag, phase portraits).
  python scripts/make_figures.py --ckpt outputs/baseline/checkpoint.pt"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.solver import StackelbergSolver
from src.visualize import plot_economic


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or Path(a.ckpt).parent / "figures")
    stats = plot_economic(StackelbergSolver.from_checkpoint(a.ckpt), out)
    (out / "economic_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats["phase"]), "\nlead-lag peak:", stats["lead_lag"]["peak_lag"], "saved ->", out)


if __name__ == "__main__":
    main()
