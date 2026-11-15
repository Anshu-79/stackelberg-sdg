# stackelberg_sdg

Modular PyTorch solver for a macroprudential Stackelberg stochastic differential game
(1 regulator + N banks), refactored from `ICORSI.ipynb`.

## Experiment-to-Script Mapping Table

| Experiment / Research Goal | Configuration File | Executable Script / Command | Output Artifacts Generated |
| :--- | :--- | :--- | :--- |
| **Baseline Stackelberg Equilibrium** | `configs/default.yaml` | `python scripts/train.py --config configs/default.yaml` | `outputs/baseline/` (loss curves, policy heatmaps) |
| **Scalability & $N$-Bank Profiling** | `configs/scaling.yaml` | `python scripts/benchmark_convergence.py --mode n_scaling` | `outputs/benchmarks/n_scaling_profile.png & json` |
| **Monte Carlo Convergence Rate** | `configs/scaling.yaml` | `python scripts/benchmark_convergence.py --mode mc_rate` | `outputs/benchmarks/mc_convergence_log-log.png` |
| **Systemic Contagion Stress Test** | `configs/experiments/systemic_shock.yaml` | `python scripts/run_stress_test.py --rho 0.8` | `outputs/stress_test/stress_trajectories.png` |
| **Ablation: Naive Co-Optimization** | `configs/ablations/ablation_coopt.yaml` | `python scripts/train.py --config configs/ablations/ablation_coopt.yaml` | `outputs/ablation_coopt/exploitability_failure.png` |
| **Ablation: DFP Mixing Weight $\beta$** | `configs/ablations/ablation_hyperparams.yaml` | `python scripts/train.py --config configs/ablations/ablation_hyperparams.yaml --override training.beta=0.2` | `outputs/ablation_beta_0.2/` |

Additional entries (same conventions):

| Experiment / Research Goal | Configuration File | Executable Script / Command | Output Artifacts Generated |
| :--- | :--- | :--- | :--- |
| **Ablation: full $\beta \times$ inner-steps grid** | `configs/ablations/ablation_hyperparams.yaml` | `python scripts/train.py --config configs/ablations/ablation_hyperparams.yaml --sweep` | `outputs/ablation_hyperparams/sweep_summary.png & json`, one folder per cell |
| **Ablation: contagion sweep $\rho \in \{0,.3,.6,.9\}$** | `configs/ablations/ablation_rho.yaml` | `python scripts/train.py --config configs/ablations/ablation_rho.yaml --sweep` | `outputs/ablation_rho/sweep_summary.png & json` |
| **Economic figures from a checkpoint** | – | `python scripts/make_figures.py --ckpt outputs/baseline/checkpoint.pt` | `policy_surfaces.png`, `lead_lag.png`, `phase_portraits.png`, `economic_stats.json` |
| **Standalone exploitability / plots** | – | `python scripts/evaluate.py --ckpt <ckpt> --k-br 300 [--fresh]` | `<run>/eval/` |

Every run also writes `config.yaml`, `history.json`, `exploitability.json`, `summary.json`, `diagnostics.json`, `checkpoint.pt`.
`--override a.b=c` (repeatable) edits any config key; `training.beta` is an alias of `fictitious_play.beta`.
`out_dir` templates such as `outputs/ablation_beta_{fictitious_play.beta}` are filled from the final config.
Add `--quick` to `benchmark_convergence.py` for a smoke test. `pytest tests -q` runs the 20 tests.

## What each experiment measures
* **n_scaling**: s/epoch, peak RSS growth (or CUDA peak) and parameter count for `independent` vs `shared`, one fresh process per cell.
  One epoch of `independent` mode performs N bank updates, `shared` performs one, so the comparison is cost per *epoch of the respective algorithm*.
* **mc_rate**: for B in {64,256,1024,4096}, RMSE (50 replications) of the Monte Carlo estimates of the value Y0 (average bank payoff) and E[mean_i X_T]
  of a trained policy vs. a B_ref=262144 reference (slope fit vs. the -1/2 line). With a `bsde_adjoint` checkpoint the learned Y0 parameter error is also reported.
* **stress test**: policies trained at rho=0; at t = `shock_time` the common-noise correlation jumps to `--rho`. Same random numbers for both arms.
  NB: raising rho *reduces* cross-sectional Var(X) (less idiosyncratic noise), so contagion shows up in the spread/tail of the mean capital X̄ across paths and in worst-bank capital.
* **coopt ablation**: joint loss `J_reg - mean_i J_i`, nothing detached. Reported: Nash exploitability and the *Stackelberg gap* =
  J_reg after freezing the regulator and letting banks best-respond selfishly minus J_reg under co-optimisation.
* **sweeps**: final exploitability, Var(X_T), regulator-loss oscillation (std of first differences, last half) and exploitability oscillation.

## Layout / extension points
| file | role | extend by |
|---|---|---|
| `dynamics.py` | SDE drift/diffusion, Euler-Maruyama, `simulate()` rollout | `@DYNAMICS.register("name")` |
| `objectives.py` | running/terminal costs and rewards | `@OBJECTIVES.register("name")` |
| `networks.py` | `PolicyNet`, `RegulatorNet`, `AdjBSDEHead`, independent/shared bank policies | `@POLICIES.register("mode")` |
| `solver.py` | DFP inner loop + leader outer loop, Polyak mixing | – |
| `metrics.py` | Nash exploitability, symmetry dispersion, `collect_paths` | – |
| `experiments.py` | single run / grid sweep / summary metrics | – |
| `benchmarks.py` | N-scaling profiler, Monte Carlo rate | – |

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
