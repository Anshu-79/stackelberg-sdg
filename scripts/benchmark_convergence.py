"""python scripts/benchmark_convergence.py --mode {n_scaling,mc_rate,all} [--quick]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.benchmarks import run_n_scaling, run_mc_rate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["n_scaling", "mc_rate", "all"], default="all")
    ap.add_argument("--config", default="configs/scaling.yaml")
    ap.add_argument("--out", default="outputs/benchmarks")
    ap.add_argument("--ckpt", default=None, help="mc_rate: evaluate this checkpoint instead of training")
    ap.add_argument("--quick", action="store_true", help="tiny sizes for smoke testing")
    ap.add_argument("--no-isolate", action="store_true", help="n_scaling: profile in-process (RSS then not per-config)")
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args()
    ov = list(a.override)
    if a.quick:
        ov += ["scaling.n_values=[10,25]", "scaling.bench_epochs=1", "mc.reps=10", "mc.ref_batch=32768",
               "mc.train_epochs=5"]
    cfg = load_config(a.config, ov)
    if a.mode in ("n_scaling", "all"):
        run_n_scaling(cfg, a.out, isolate=not a.no_isolate)
    if a.mode in ("mc_rate", "all"):
        run_mc_rate(cfg, a.out, ckpt=a.ckpt)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
