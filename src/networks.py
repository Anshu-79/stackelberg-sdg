"""Policy networks: PolicyNet, RegulatorNet, AdjBSDEHead, independent / shared bank policies."""
from __future__ import annotations
import torch
import torch.nn as nn
from .registry import POLICIES, REGULATORS

ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


class PolicyNet(nn.Module):
    """MLP on concat([t, state]). Works for [B,k] and [B,N,k] inputs."""
    def __init__(self, in_dim, out_dim, hidden=32, depth=2, activation="relu"):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), ACTS[activation]()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, state):
        return self.net(torch.cat([t, state], dim=-1))


class AdjBSDEHead(PolicyNet):
    """Dual-head network returning (u, Z, Z0): control, idiosyncratic and common adjoint."""
    def __init__(self, in_dim, **kw):
        super().__init__(in_dim, 3, **kw)

    def forward(self, t, state):
        o = super().forward(t, state)
        return o[..., 0], o[..., 1], o[..., 2]


@REGULATORS.register("mlp")
class RegulatorNet(nn.Module):
    """alpha_t = a_min + (a_max - a_min) * sigmoid(NN(t, mean X, Var X))."""
    def __init__(self, hidden=32, depth=2, activation="relu", alpha_min=0.0, alpha_max=1.0):
        super().__init__()
        self.net = PolicyNet(3, 1, hidden, depth, activation)
        self.a_min, self.a_max = alpha_min, alpha_max

    def forward(self, t, mean_X, var_X):
        raw = self.net(t, torch.cat([mean_X, var_X], dim=-1))
        return self.a_min + (self.a_max - self.a_min) * torch.sigmoid(raw)


class _BankPolicy(nn.Module):
    """forward(t, X, mean_X, alpha, idx=None) -> (U, Z, Z0), each [B,N] (or [B,len(idx)]).
    Z, Z0 are None unless bsde=True (then each net is an AdjBSDEHead-style 3-output MLP)."""
    bsde = False

    def _unpack(self, out):
        if self.bsde:
            return out[..., 0], out[..., 1], out[..., 2]
        return out[..., 0], None, None

    @property
    def out_dim(self):
        return 3 if self.bsde else 1


@POLICIES.register("independent")
class IndependentBankPolicy(_BankPolicy):
    def __init__(self, n_banks, hidden=32, depth=2, activation="relu", bsde=False):
        super().__init__()
        self.n_banks, self.bsde = n_banks, bsde
        self.nets = nn.ModuleList([PolicyNet(4, self.out_dim, hidden, depth, activation)
                                   for _ in range(n_banks)])
        self.y0 = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(n_banks)]) if bsde else None

    def forward(self, t, X, mean_X, alpha, idx=None):
        idx = list(range(self.n_banks)) if idx is None else list(idx)
        outs = [self.nets[i](t, torch.cat([X[:, i:i + 1], mean_X, alpha], -1)) for i in idx]
        return self._unpack(torch.stack(outs, 1))

    def param_groups(self, i):
        return list(self.nets[i].parameters()) + ([self.y0[i]] if self.bsde else [])

    def n_optimizers(self):
        return self.n_banks

    def optimizer_index(self, i):
        return i

    def y0_vector(self):
        return torch.stack(list(self.y0)) if self.bsde else None


@POLICIES.register("shared")
class SharedBankPolicy(_BankPolicy):
    """Single network NN(t, X^i, Xbar, alpha) evaluated on all banks (O(1) params in N)."""
    def __init__(self, n_banks, hidden=32, depth=2, activation="relu", bsde=False):
        super().__init__()
        self.n_banks, self.bsde = n_banks, bsde
        self.net = PolicyNet(4, self.out_dim, hidden, depth, activation)
        self.y0 = nn.Parameter(torch.zeros(())) if bsde else None

    def forward(self, t, X, mean_X, alpha, idx=None):
        Xs = X if idx is None else X[:, list(idx)]
        n = Xs.shape[1]
        state = torch.cat([Xs.unsqueeze(-1),
                           mean_X.unsqueeze(1).expand(-1, n, -1),
                           alpha.unsqueeze(1).expand(-1, n, -1)], -1)
        return self._unpack(self.net(t.unsqueeze(1).expand(-1, n, -1), state))

    def param_groups(self, i=None):
        return list(self.parameters())

    def n_optimizers(self):
        return 1

    def optimizer_index(self, i):
        return 0

    def y0_vector(self):
        return self.y0.expand(self.n_banks) if self.bsde else None


@POLICIES.register("bsde_adjoint")
def _bsde_policy(n_banks, share_parameters=False, **kw):
    cls = SharedBankPolicy if share_parameters else IndependentBankPolicy
    return cls(n_banks, bsde=True, **kw)


def build_bank_policy(cfg, n_banks):
    kw = dict(hidden=cfg.hidden, depth=cfg.depth, activation=cfg.activation)
    if cfg.mode == "bsde_adjoint":
        kw["share_parameters"] = cfg.get("share_parameters", False)
    return POLICIES.build(cfg.mode, n_banks=n_banks, **kw)


def mix_outputs(live_out, opp_out, idx):
    """Columns `idx` come from `live_out` (with grad); all others from frozen `opp_out`."""
    idx = list(idx)
    res = []
    for l, o in zip(live_out, opp_out):
        if l is None:
            res.append(None)
            continue
        m = o.detach().clone()
        m[:, idx] = l
        res.append(m)
    return tuple(res)
