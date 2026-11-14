"""Bi-level Stackelberg trainer with Deep Fictitious Play for the followers."""
from __future__ import annotations
import copy, json, random
from pathlib import Path
import torch
import yaml

from .config import Cfg
from .registry import DYNAMICS, OBJECTIVES, REGULATORS
from .dynamics import simulate
from .networks import build_bank_policy, mix_outputs
from .metrics import exploitability, rollout_diagnostics


class StackelbergSolver:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        dev = cfg.training.device
        self.dyn = DYNAMICS.build(cfg.dynamics.name, **cfg.game, device=dev)
        self.obj = OBJECTIVES.build(cfg.objectives.name, **{k: v for k, v in cfg.objectives.items() if k != "name"})
        rc = cfg.regulator
        self.regulator = REGULATORS.build(rc.name, hidden=rc.hidden, depth=rc.depth, activation=rc.activation,
                                          alpha_min=rc.alpha_min, alpha_max=rc.alpha_max).to(dev)
        self.N = cfg.game.n_banks
        self.banks = build_bank_policy(cfg.banks, self.N).to(dev)
        self.opp = copy.deepcopy(self.banks)          # frozen opponent policy (EMA / Polyak mixture)
        for p in self.opp.parameters():
            p.requires_grad_(False)
        self.opt_reg = torch.optim.Adam(self.regulator.parameters(), lr=rc.lr)
        self.opt_banks = [torch.optim.Adam(self.banks.param_groups(i), lr=cfg.banks.lr)
                          for i in range(self.banks.n_optimizers())]
        self.history, self.expl_curve = [], []
        self.batch = cfg.training.batch_size

    # ---------------- helpers ----------------
    def _polyak(self):
        beta = self.cfg.fictitious_play.beta
        with torch.no_grad():
            for po, pl in zip(self.opp.parameters(), self.banks.parameters()):
                po.mul_(1.0 - beta).add_(pl.detach(), alpha=beta)

    def _bank_schedule(self):
        fp = self.cfg.fictitious_play
        if self.banks.n_optimizers() == 1:            # shared: random subset of simultaneous deviators
            k = min(fp.shared_active_banks, self.N)
            return [sorted(random.sample(range(self.N), k))]
        return [[i] for i in range(self.N)]

    # ---------------- updates ----------------
    def bank_step(self, idx):
        live, opp = self.banks, self.opp
        w = self.cfg.fictitious_play.bsde_weight

        def bank_fn(t, X, m, a):
            with torch.no_grad():
                o = opp(t, X, m, a)
            return mix_outputs(live(t, X, m, a, idx=idx), o, idx)

        y0 = None
        if live.bsde:
            y0 = opp.y0_vector().detach().clone()
            y0[idx] = live.y0_vector()[idx]
        opt = self.opt_banks[live.optimizer_index(idx[0])]
        opt.zero_grad(set_to_none=True)
        out = simulate(self.dyn, self.obj, self.regulator, bank_fn, self.batch, y0=y0,
                       leader_grad=False, detach_mean=True)
        loss = -out["payoffs"][idx].mean()            # banks maximise
        if out["bsde_residual"] is not None:
            loss = loss + w * out["bsde_residual"][idx].mean()
        loss.backward()
        opt.step()
        self._polyak()
        return loss.item()

    def leader_step(self):
        lc = self.cfg.leader
        diff = lc.differentiate_through_followers

        def bank_fn(t, X, m, a):
            if diff:
                return self.banks(t, X, m, a)
            with torch.no_grad():
                return self.banks(t, X, m, a)

        self.opt_reg.zero_grad(set_to_none=True)
        out = simulate(self.dyn, self.obj, self.regulator, bank_fn, self.batch,
                       leader_grad=True, detach_mean=lc.detach_mean)
        out["reg_loss"].backward()
        self.opt_reg.step()
        if diff:
            for p in self.banks.parameters():
                p.grad = None
        return out["reg_loss"].item(), out["payoffs"].mean().item()

    # ---------------- evaluation ----------------
    def evaluate_exploitability(self, fresh=None, k_br=None, n_banks=None):
        ec = self.cfg.exploitability
        n = min(n_banks or ec.n_eval_banks, self.N)
        idx = sorted(random.Random(self.cfg.training.seed).sample(range(self.N), n))
        return exploitability(self.dyn, self.obj, self.regulator, self.banks, bank_idx=idx,
                              k_br=k_br or ec.k_br, lr=ec.lr, train_batch=self.batch,
                              eval_batch=ec.eval_batch, fresh=ec.fresh if fresh is None else fresh,
                              detach_mean=True)

    def diagnostics(self, **kw):
        return rollout_diagnostics(self.dyn, self.obj, self.regulator, self.banks, **kw)

    # ---------------- training ----------------
    def train(self, log=print):
        tc, fp = self.cfg.training, self.cfg.fictitious_play
        for epoch in range(1, tc.epochs + 1):
            for _ in range(fp.inner_steps_per_epoch):
                for idx in self._bank_schedule():
                    self.bank_step(idx)
            reg_loss, avg_payoff = self.leader_step()
            self.history.append({"epoch": epoch, "reg_loss": reg_loss, "avg_bank_payoff": avg_payoff})
            msg = None
            if epoch % tc.log_every == 0 or epoch == tc.epochs:
                msg = f"Epoch {epoch:03d} | Regulator Systemic Risk: {reg_loss:.4f} | Avg Bank Payoff: {avg_payoff:.4f}"
            if tc.eval_every > 0 and (epoch % tc.eval_every == 0 or epoch == tc.epochs):
                e = self.evaluate_exploitability()
                self.expl_curve.append({"epoch": epoch, "mean": e["mean"], "per_bank": e["per_bank"]})
                msg = (msg or f"Epoch {epoch:03d}") + f" | Exploitability: {e['mean']:.5f}"
            if msg:
                log(msg)
        return self.history

    # ---------------- io ----------------
    def save(self, out_dir):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(self.cfg.to_dict()))
        (out / "history.json").write_text(json.dumps(self.history))
        (out / "exploitability.json").write_text(json.dumps(self.expl_curve))
        torch.save({"cfg": self.cfg.to_dict(), "regulator": self.regulator.state_dict(),
                    "banks": self.banks.state_dict(), "opp": self.opp.state_dict()}, out / "checkpoint.pt")

    @classmethod
    def from_checkpoint(cls, path, overrides=None):
        from .config import apply_overrides
        ck = torch.load(path, map_location="cpu", weights_only=False)
        solver = cls(Cfg(apply_overrides(ck["cfg"], overrides)))
        solver.regulator.load_state_dict(ck["regulator"])
        solver.banks.load_state_dict(ck["banks"])
        solver.opp.load_state_dict(ck["opp"])
        return solver
