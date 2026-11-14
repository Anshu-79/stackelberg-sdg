# stackelberg_sdg

Modular PyTorch solver for a macroprudential Stackelberg stochastic differential game
(1 regulator + N banks), refactored from `ICORSI.ipynb`.

## Quick start
```bash
pip install torch pyyaml matplotlib pytest
python scripts/train.py --config configs/default.yaml                      # reproduces the notebook setup
python scripts/train.py --config configs/experiments/shared_n50.yaml
python scripts/train.py --config configs/experiments/systemic_shock.yaml
python scripts/train.py banks.mode=bsde_adjoint game.rho=0.3 training.epochs=100   # dot-notation overrides
python scripts/evaluate.py --ckpt runs/default/checkpoint.pt --k-br 200 --fresh
pytest tests -q
```

## Layout / extension points
| file | role | extend by |
|---|---|---|
| `dynamics.py` | SDE drift/diffusion, Euler-Maruyama, `simulate()` rollout | `@DYNAMICS.register("name")` |
| `objectives.py` | running/terminal costs and rewards | `@OBJECTIVES.register("name")` |
| `networks.py` | `PolicyNet`, `RegulatorNet`, `AdjBSDEHead`, independent/shared bank policies | `@POLICIES.register("mode")` |
| `solver.py` | DFP inner loop + leader outer loop, Polyak mixing | – |
| `metrics.py` | Nash exploitability, symmetry dispersion, drift stats | – |

## Notes on algorithmic choices
* **DFP isolation**: bank *i*'s update uses its live network for column *i* and a frozen *opponent copy*
  (`solver.opp`) for all others. `fictitious_play.beta` Polyak-averages that copy
  (`beta=1` = latest parameters, identical to the notebook).
* **Shared mode**: `shared_active_banks` banks deviate simultaneously per step; everyone else uses the frozen copy.
* **Leader pass**: by default followers' controls are detached (as in the notebook), so the leader
  ignores follower reactions. Set `leader.differentiate_through_followers=true` to differentiate through
  `u^i(t, X, Xbar, alpha)` (closer to a true Stackelberg leader; follower parameters are still frozen).
* **Exploitability**: best response = max over {u^i, Adam-ascent checkpoints} evaluated on common
  random numbers, hence E_i >= 0 exactly; it is a lower bound on the true exploitability.
* **bsde_adjoint**: policy heads `(u, Z, Z0)` plus learnable `Y_0`; the forward adjoint
  `Y_{t+dt} = Y_t - f dt + Z dW^i + Z0 dW^0` is penalised at T by `E|Y_T - g(X_T)|^2` (weight `bsde_weight`)
  as an auxiliary loss on top of the policy-gradient payoff.
