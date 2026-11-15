"""Nash exploitability, symmetry dispersion, drift statistics."""
from __future__ import annotations
import copy
import torch
from .dynamics import simulate
from .networks import mix_outputs


def _eval_payoffs(dyn, obj, regulator, fn, batch, seed, detach_mean=True):
    """Payoffs on common random numbers (fixed seed) so that differences are paired."""
    gen = torch.Generator(device=dyn.device).manual_seed(seed)
    with torch.no_grad():
        return simulate(dyn, obj, regulator, fn, batch, generator=gen, detach_mean=detach_mean)["payoffs"]


def exploitability(dyn, obj, regulator, policy, *, bank_idx, k_br=50, lr=1e-3,
                   train_batch=64, eval_batch=2048, seed=1234, fresh=False, detach_mean=True):
    """E_i = J_i^BR(u^-i, alpha) - J_i(u^i; u^-i, alpha) >= 0.

    alpha and all u^-i are frozen. A test copy of bank i's network (cloned or re-initialised)
    does k_br Adam ascent steps on J_i. J_i^BR is the best value over {u^i itself, checkpoints
    of the ascent} evaluated on common random numbers, so E_i >= 0 by construction."""
    base_fn = lambda t, X, m, a: policy(t, X, m, a)
    J = _eval_payoffs(dyn, obj, regulator, base_fn, eval_batch, seed, detach_mean)
    J_br = J.clone()
    chk = max(1, k_br // 4)
    for i in bank_idx:
        test = copy.deepcopy(policy)
        if fresh:
            for m in test.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        for p in test.parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(test.param_groups(i), lr=lr)

        def fn(t, X, m, a, test=test, i=i):
            with torch.no_grad():
                o = policy(t, X, m, a)
            return mix_outputs(test(t, X, m, a, idx=[i]), o, [i])

        for k in range(k_br):
            opt.zero_grad(set_to_none=True)
            out = simulate(dyn, obj, regulator, fn, train_batch, detach_mean=detach_mean)
            (-out["payoffs"][i]).backward()
            opt.step()
            if (k + 1) % chk == 0 or k == k_br - 1:
                J_br[i] = torch.maximum(J_br[i], _eval_payoffs(dyn, obj, regulator, fn, eval_batch, seed, detach_mean)[i])
    idx = list(bank_idx)
    E = (J_br - J)[idx]
    return {"per_bank": E.tolist(), "mean": E.mean().item(), "bank_idx": idx,
            "J": J[idx].tolist(), "J_br": J_br[idx].tolist()}


def rollout_diagnostics(dyn, obj, regulator, policy, batch=2048, seed=0, detach_mean=True):
    gen = torch.Generator(device=dyn.device).manual_seed(seed)
    fn = lambda t, X, m, a: policy(t, X, m, a)
    with torch.no_grad():
        out = simulate(dyn, obj, regulator, fn, batch, generator=gen, record=True, detach_mean=detach_mean)
    r = out["rec"]
    U_bank = r["U"].mean(0)                       # time-avg control per bank [N]
    return {
        "t": (torch.arange(dyn.n_steps) * dyn.dt).tolist(),
        "alpha": r["alpha"].tolist(), "mean_X": r["mean_X"].tolist(), "var_X": r["var_X"].tolist(),
        "drift_mean": r["drift_mean"].tolist(), "drift_std": r["drift_std"].tolist(),
        "U_bank": U_bank.tolist(), "payoff_bank": out["payoffs"].tolist(),
        "symmetry_dispersion_u": U_bank.std().item(),
        "symmetry_dispersion_payoff": out["payoffs"].std().item(),
        "reg_loss": out["reg_loss"].item(),
        "var_XT": out["X_T"].var(dim=1).mean().item(), "mean_XT": out["X_T"].mean().item(),
    }


def collect_paths(dyn, obj, regulator, policy, batch=512, seed=0, detach_mean=True):
    """Roll out and return numpy arrays: t[S], mean_paths/var_paths/alpha_paths[S,B], X_full/U_full[S,B,N], X_T[B,N]."""
    gen = torch.Generator(device=dyn.device).manual_seed(seed)
    fn = lambda t, X, m, a: policy(t, X, m, a)
    with torch.no_grad():
        out = simulate(dyn, obj, regulator, fn, batch, generator=gen, record=True, detach_mean=detach_mean)
    P = {k: out["rec"][k].cpu().numpy() for k in ("mean_paths", "var_paths", "alpha_paths", "X_full", "U_full")}
    P["t"] = [k * dyn.dt for k in range(dyn.n_steps)]
    P["t"] = __import__("numpy").array(P["t"])
    P["X_T"] = out["X_T"].cpu().numpy()
    return P
