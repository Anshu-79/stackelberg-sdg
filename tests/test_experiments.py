import copy, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pytest, torch
from src.config import load_config, apply_overrides
from src.solver import StackelbergSolver
from src.experiments import resolve_path, run_sweep
from src.metrics import collect_paths
from src.benchmarks import mc_chunks, profile_worker

ROOT = Path(__file__).resolve().parents[1]
FAST = ["game.n_banks=4", "game.n_steps=5", "training.batch_size=16", "exploitability.k_br=3",
        "exploitability.eval_batch=64", "exploitability.n_eval_banks=1", "training.eval_every=0"]


def make(*ov, cfg="configs/default.yaml"):
    torch.manual_seed(0); random.seed(0)
    return StackelbergSolver(load_config(ROOT / cfg, FAST + list(ov)))


def test_coopt_updates_everything_jointly():
    s = make("training.mode=coopt")
    r0 = [p.clone() for p in s.regulator.parameters()]; b0 = [p.clone() for p in s.banks.parameters()]
    s.train_epoch()
    assert any(not torch.equal(a, b) for a, b in zip(r0, s.regulator.parameters()))
    assert sum(not torch.equal(a, b) for a, b in zip(b0, s.banks.parameters())) > 0


def test_time_varying_rho():
    s = make()
    d = copy.copy(s.dyn); d.shock_time, d.shock_rho = 0.5, 1.0
    assert d.rho_at(0.2) == 0.0 and d.rho_at(0.6) == 1.0
    dW0, dWi = d.noise(6)
    X = torch.zeros(6, 4)
    Xn = d.step(X, torch.zeros_like(X), dW0, dWi, t=0.7)
    assert torch.allclose(Xn, d.sigma * dW0.expand_as(Xn))
    assert d.noise(6)[0].abs().sum() > 0        # common shock sampled once a shock is configured


def test_alias_and_path_template():
    cfg = load_config(ROOT / "configs/ablations/ablation_hyperparams.yaml", ["training.beta=0.2"])
    assert cfg.fictitious_play.beta == 0.2
    assert resolve_path(cfg.training.out_dir, cfg) == "outputs/ablation_beta_0.2"


def test_collect_paths_shapes():
    s = make()
    P = collect_paths(s.dyn, s.obj, s.regulator, s.banks, batch=8)
    assert P["X_full"].shape == (5, 8, 4) and P["alpha_paths"].shape == (5, 8) and P["t"].shape == (5,)


def test_mc_chunks_and_profile():
    s = make()
    y, x = mc_chunks(s, 64, 0, 32)
    assert y.shape == (2,)
    r = profile_worker(s.cfg.to_dict(), 1)
    assert r["time_per_epoch"] > 0 and r["n_params"] > 0


def test_sweep_runner(tmp_path):
    cfg = load_config(ROOT / "configs/ablations/ablation_rho.yaml", FAST + ["training.epochs=2", "training.eval_every=2",
                      f"sweep_out_dir={tmp_path}/rho_{{game.rho}}"])
    cfg["sweep"] = {"game.rho": [0.0, 0.5]}
    cfg.training.out_dir = cfg.sweep_out_dir
    res = run_sweep(cfg, log=lambda *a: None)
    assert len(res) == 2 and (tmp_path / "sweep_summary.json").exists()
