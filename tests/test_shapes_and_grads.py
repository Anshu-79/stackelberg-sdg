import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest, torch
from src.config import load_config
from src.solver import StackelbergSolver
from src.dynamics import simulate

CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
FAST = ["game.n_banks=4", "game.n_steps=5", "training.batch_size=16", "exploitability.k_br=4",
        "exploitability.eval_batch=64", "exploitability.n_eval_banks=2"]


def make(*ov, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    return StackelbergSolver(load_config(CFG, FAST + list(ov)))


def has_grad(params):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)


@pytest.mark.parametrize("mode", ["independent", "shared", "bsde_adjoint"])
def test_shapes(mode):
    s = make(f"banks.mode={mode}")
    out = simulate(s.dyn, s.obj, s.regulator, lambda t, X, m, a: s.banks(t, X, m, a), 16)
    assert out["payoffs"].shape == (4,) and out["reg_loss"].ndim == 0
    U, Z, Z0 = s.banks(torch.zeros(16, 1), torch.randn(16, 4), torch.zeros(16, 1), torch.zeros(16, 1))
    assert U.shape == (16, 4) and ((Z is not None) == (mode == "bsde_adjoint"))


def test_dfp_gradient_isolation_independent():
    s = make("banks.mode=independent")
    s.bank_step([2])
    # only bank 2's optimiser step happened; verify grads: bank 2 nonzero, others None, regulator None
    assert has_grad(s.banks.nets[2].parameters())
    for j in (0, 1, 3):
        assert all(p.grad is None for p in s.banks.nets[j].parameters())
    assert all(p.grad is None for p in s.regulator.parameters())
    assert all(p.grad is None for p in s.opp.parameters())


def test_bank_step_changes_only_active_bank():
    s = make("banks.mode=independent")
    before = [[p.clone() for p in n.parameters()] for n in s.banks.nets]
    s.bank_step([1])
    for j, n in enumerate(s.banks.nets):
        changed = any(not torch.equal(a, b) for a, b in zip(before[j], n.parameters()))
        assert changed == (j == 1)


def test_leader_pass_isolation():
    s = make()
    s.leader_step()   # default: followers detached in leader pass
    assert has_grad(s.regulator.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in s.banks.parameters())


def test_leader_differentiates_through_followers_flag():
    s = make("leader.differentiate_through_followers=true")
    p0 = [p.clone() for p in s.banks.parameters()]
    s.leader_step()
    assert all(torch.equal(a, b) for a, b in zip(p0, s.banks.parameters()))  # banks never updated by leader
    assert all(p.grad is None for p in s.banks.parameters())                  # grads cleaned


def test_shared_mode_grads_and_param_count():
    s = make("banks.mode=shared", "fictitious_play.shared_active_banks=2")
    s.bank_step([0, 3])
    assert has_grad(s.banks.parameters()) and all(p.grad is None for p in s.opp.parameters())
    n_shared = sum(p.numel() for p in s.banks.parameters())
    n_ind = sum(p.numel() for p in make("banks.mode=independent").banks.parameters())
    assert n_ind == 4 * n_shared


def test_polyak_beta():
    s = make("fictitious_play.beta=0.5")
    s.bank_step([0])
    for po, pl in zip(s.opp.nets[0].parameters(), s.banks.nets[0].parameters()):
        assert not torch.equal(po, pl)          # opp lags live params when beta < 1
    s2 = make("fictitious_play.beta=1.0")
    s2.bank_step([0])
    for po, pl in zip(s2.opp.parameters(), s2.banks.parameters()):
        assert torch.equal(po, pl)


def test_bsde_step_runs():
    s = make("banks.mode=bsde_adjoint", "game.rho=0.5")
    assert isinstance(s.bank_step([1]), float)
    assert has_grad([s.banks.y0[1]])


def test_common_shock_correlation():
    s = make("game.rho=1.0")
    dW0, dWi = s.dyn.noise(8)
    X = torch.zeros(8, 4)
    Xn = s.dyn.step(X, torch.zeros_like(X), dW0, dWi)
    assert torch.allclose(Xn, s.dyn.sigma * dW0.expand_as(Xn))   # rho=1 -> purely common shock


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_exploitability_nonnegative_and_freezes_policy(mode):
    s = make(f"banks.mode={mode}")
    snap = [p.clone() for p in s.banks.parameters()]
    e = s.evaluate_exploitability()
    assert all(x >= 0 for x in e["per_bank"])
    assert all(torch.equal(a, b) for a, b in zip(snap, s.banks.parameters()))   # original untouched


def test_exploitability_detects_non_equilibrium():
    s = make("exploitability.k_br=60", "exploitability.lr=1e-2")
    assert s.evaluate_exploitability()["mean"] > 0
