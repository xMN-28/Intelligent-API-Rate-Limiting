# Reinforcement learning traffic control

The current project is **Reinforcement Learning-Based API Rate Control**. The CLI
and Python package are `rlrc`; environment configuration uses `RLRC_`. The original
interfaces are preserved on `codex/legacy-ewma-aimd`. Historical measurement files
retain their original source paths and hashes. Resumable checkpoints and raw traces
are published with [v0.2.1](https://github.com/xMN-28/Reinforcement-Learning-Based-API-Rate-Control/releases/tag/v0.2.1).

The conventional Redis limiters and EWMA/AIMD controller remain intact. The upgrade
adds a real PPO training system, deterministic trained policy deployment, an
optional overload predictor, safety guards, and matched simulator/HTTP experiments.
Measured outcomes are in [the upgrade report](../results/rl-upgrade/REPORT.md).

The editable [architecture diagram](rl-architecture.mmd) separates offline training
from live admission. Models never execute on individual admission checks; the
original Redis scripts remain the enforcement boundary.

## Install

```powershell
uv sync --frozen --extra dev --extra ml
```

The pinned stack is PyTorch 2.8.0, Gymnasium 1.2.0, Stable-Baselines3 2.7.0,
scikit-learn 1.7.2, TensorBoard 2.20.0 and the existing NumPy/Python environment.
Windows uses the official PyTorch CUDA 12.8 index; Linux uses the official CPU
index for smaller CI/runtime installs. CUDA execution was checked on the detected
RTX 4070 SUPER. A CUDA wheel can still execute on CPU when CUDA is unavailable.
Only trusted project-trained PPO checkpoints and joblib files should be loaded.

## Run the trained gateway

The repository intentionally includes small deployment exports under
`assets/models/`; large training logs and trainable checkpoints stay ignored.

```powershell
$env:RLRC_CONFIG = 'configs/rl.yaml'
uv run rlrc serve backend
# In a separate terminal with the same RLRC_CONFIG:
uv run rlrc serve gateway
```

The default RL profile is `rl_safe`, 20 initial RPS, a 10–120 RPS range, a 300ms
latency target, four 100ms workers, an eight-request queue and a 64-request gateway
in-flight ceiling. Its configuration must match the model's saved normalization.
The trained variants require a one-second control interval; changing it triggers
explicit fallback because action timing and forecast horizons would otherwise differ.

Available policy names:

| Policy | Control |
|---|---|
| `none` | No rate quota; fixed in-flight ceiling remains |
| `token_bucket` | Existing static atomic Token Bucket |
| `sliding_window` | Existing exact Sliding Window Log |
| `adaptive` | Original EWMA/AIMD feedback policy |
| `rl` | Deterministic PPO actor with hard rate bounds |
| `rl_safe` | Same actor with explicit safety interventions |
| `predictive` | Original AIMD plus predicted-risk backoff |
| `hybrid` | PPO trained with risk input, plus safety |

Backend-aware PPO uses `rl`/`rl_safe` with `ml.backend_telemetry: true` and the
matching 19-feature model. It is reported separately from gateway-only PPO. For
`hybrid`, use the `ppo-prediction` actor and the optional predictor artifact. The
predictor needs the ML extra; plain PPO deployment only needs NumPy and the normal
application dependencies.

For Docker with the RL profile, set all three overrides because the existing
Compose policy/rate defaults intentionally preserve the original baseline:

```powershell
$env:RLRC_CONFIG = 'configs/rl.yaml'
$env:RLRC_POLICY = 'rl_safe'
$env:RLRC_RATE = '20'
docker compose up --build -d
```

The Docker image copies the small deployment exports. `WITH_ML=true` is an optional
build argument for an image that also needs sklearn/training dependencies. A local
complete Docker/Grafana run has not been verified while the engine is unavailable.

For an explicitly configured predictive/hybrid profile, install the optional stack
into the gateway image before bringing up the already-built services:

```powershell
docker compose build --build-arg WITH_ML=true gateway
docker compose up -d --no-build
```

The profile must point to `assets/models/overload-predictor/predictor.joblib` and,
for hybrid, `assets/models/ppo-prediction`. Rebuilding later without the build
argument removes those optional dependencies.

## Simulator

`Simulator` is a request-level discrete-event model. One Gymnasium transition is
one simulated second; no wall-clock sleep or real HTTP is involved in training.
Events explicitly represent backend service completion, response delivery, queue
overflow, and gateway timeout. Timed-out backend work can continue to hold workers
or queue slots, as in the real implementation. Capacity shrink drains existing
work rather than cancelling it.

Global/client token balances are settled before budget changes and clamped to new
capacities. Exact rolling logs are available for the static Sliding Window
comparator. The same fixed in-flight ceiling and equal client quotas apply.
Requests and their service variates are generated before controller actions so a
policy cannot change the exogenous trace by admitting a different number of calls.

Training varies worker counts, service means, uniform/lognormal service times,
offered load, bursts, ramps, sudden/gradual degradation, recovery, and client
weights. Optional Poisson arrivals are supported. Real fidelity tests use the
regular-arrival, uniform-jitter mode that matches the existing backend/generator.
The default training file is `configs/training.yaml`.

`reset(seed=...)` seeds the episode generator; `step` returns the normalized
observation, scalar reward, `terminated=False`, and time-limit truncation at the
episode boundary. The environment passes Stable-Baselines3's checker. Independent
backend-telemetry dropout does not perturb exogenous episode-seed generation.

Fidelity was tested with real HTTP at 10/20/30/40/60 RPS and a separate worker-drop
and recovery trace. Nominal simulator throughput agrees within about 2% on that
sweep; successful p95 is approximately 11–34ms too low. Fixed 5ms transport overhead,
OS scheduling, FIFO approximation and uniform/lognormal service distributions are
limitations. This is not a calibrated digital twin of an arbitrary upstream.

## Observation and action definitions

The shared `FeatureBuilder` is used by training and real inference. It never sees
future demand/capacity, a scenario phase index, the worker schedule, or an oracle
maximum sustainable throughput.

The 16 gateway-only features are budget, offered/admitted/completed/goodput RPS,
EWMA attempt latency, recent p90 attempt latency, failure fraction, timeout
fraction, rejection fraction, gateway in-flight fraction, prior relative rate
change, demand trend, latency trend, sample volume and a has-samples flag.

RPS features divide by the saved maximum rate, latency features by four times the
latency target, in-flight count by its fixed ceiling, and sample count by 100.
Features are float32 and clipped to `[-1, 4]`; non-finite telemetry raises an
explicit inference failure rather than being clipped into a valid state. No
unsaved running-normalization state is needed. A backend variant appends queue
utilization, busy-worker utilization and an availability mask; a prediction variant
appends risk probability. The learned EWMA is fixed at alpha 0.2 in the feature
schema; it is distinct from the conventional controller's configurable smoothing.

Model input uses only observations available at decision time. Its recent goodput
feature is an upstream-duration estimate; definitive benchmark goodput uses raw
client-observed latency, including network/gateway effects. That distinction is
important for simulation transfer and for interpreting inference features.

The six deterministic discrete actions multiply the current budget by
`[0.80, 0.90, 0.95, 1.00, 1.05, 1.10]`, then enforce the common rate bounds. The
training/control interval is one second. Changing the interval changes the control
problem and requires separate validation, not an assumption of equivalent behavior.

## Reward

For one interval let `O` be offered requests, `G` on-time successful completions,
`L` late successful completions, `E` backend errors, `T` timeouts, `C` all observed
completions, and `J` Jain's successful-throughput fairness score. Define
`D = max(1, O, C)` and relative budget change `u = new_rate / old_rate - 1`.

```text
reward = G / D
       - 0.70 × L / D
       - 2.00 × (E + T) / D
       - 0.03 × rejected / max(1, O)
       - 0.02 × abs(u)
       - 0.02 × (1 - J)          [equal-demand clients with >=6 successes only]
```

Completions are delayed events; merely admitting requests earns no reward.
Reject-all loses the rejection penalty and foregoes positive useful-work reward;
excess admission incurs late-work and failure penalties. No-demand hold gets zero.
Individual components are logged. Unit tests compare useful service with
reject-all and overload, but they do not prove that no reward exploit exists.
Independent goodput/error/latency metrics, not reward alone, decide comparisons.

## PPO training and reproducibility

```powershell
uv run --extra ml rlrc ml hardware --output results/ml-runs/hardware-new.json
uv run --extra ml rlrc ml train --output models/new-gateway --steps 131072 --seed 42 --n-envs 4 --vec subproc --device cpu --sim-config configs/training.yaml
uv run --extra ml rlrc ml train --output models/new-backend --steps 131072 --seed 42 --n-envs 4 --vec subproc --device cpu --backend
uv run --extra ml rlrc ml predictor --output models/new-predictor --episodes 100
uv run --extra ml rlrc ml train --output models/new-prediction --steps 131072 --seed 42 --n-envs 4 --vec subproc --device cpu --predictor models/new-predictor/predictor.joblib
```

Selected local artifacts from the actual experiments are `models/ppo-gateway`,
`models/ppo-backend-v2`, `models/ppo-prediction-v2` and `models/overload-predictor`.
Earlier exploratory variants are retained rather than overwritten. Use new output
directories for reproduction. The benchmark driver currently uses those explicit
local model names; adjust its model mapping deliberately for a newly named run.
If the local training directories are absent, benchmark mappings use the included
deployment exports and record their actual paths. `ml sim --policies adaptive rl
rl_safe` evaluates a subset without loading the optional predictor or backend-aware
actor. The full comparison deliberately includes all variants.

Actor and value networks each have two 64-unit Tanh layers. PPO uses 256 rollout
steps per environment, batch size 256, six epochs, learning rate 0.0003, gamma 0.98,
GAE lambda 0.95 and entropy coefficient 0.01. Remaining defaults follow pinned SB3
2.7.0. Four subprocess workers were faster than serial/DummyVecEnv and CUDA on the
measured small-MLP workload. There is no reason to fill 12GB VRAM for this network.

Six fixed validation seeds 20001–20006 select the best checkpoint. Selection score
is validation goodput RPS minus `10 × backend-error/offered fraction` minus
`5 × violating-seconds / measured-seconds`. It is separate from the training reward.
The best checkpoint can precede the final training transition. Final simulator
tests use seeds 40000–40004 and HTTP tests use 81000–81002. Evaluation never trains
the actor or updates normalization.

Resume weights and optimizer state into a new output directory:

```powershell
uv run --extra ml rlrc ml train --output models/resumed --resume models/ppo-gateway/last.zip --steps 131072 --seed 42 --n-envs 4 --vec subproc --device cpu
uv run --extra ml tensorboard --logdir models/ppo-gateway/tensorboard
```

Resuming resets simulator episodes; it is not a bit-exact restoration of every
in-flight environment job. Metadata/feature/config mismatches are rejected. An
interruption saves `last.zip`, completed step count and an incomplete manifest.
Periodic checkpoints, `best.zip`, `final.zip`, metadata, validation curves and
TensorBoard events remain available. Seeding and pinned dependencies support
reproduction, not a promise of identical floating-point trajectories across hosts.

## Real inference and safety

The gateway loads a small numeric actor export once at startup. Its logits use the
same trained Linear/Tanh weights as PPO; deterministic argmax actions are checked
against SB3. This avoids importing Torch and loading CUDA libraries in the gateway.
Inference occurs in a worker thread on the control interval, never per request.

Request handlers only collect bounded counters/latencies and use the original Redis
admission scripts. A control tick builds features, evaluates the actor, validates
its action, optionally applies guards and publishes through the existing atomic
token-settlement operation. Optional backend polling happens once per tick, not
per request. Backend telemetry failure sets its availability mask and is logged.

The safety layer enforces bounds and rate-jump limits, reacts to fresh timeouts,
backs off when recent errors/latency are unhealthy and blocks increases with fewer
than five observations. Missing/corrupt models, wrong feature schema or
normalization, NaNs, invalid actions, inference errors and excessive inference
duration trigger explicit AIMD fallback. Every decision records proposed/applied
rates, action, intervention, fallback reason, publication acknowledgement,
inference/control time and process resource observations. Redis publication failure
restores local controller rate; namespace reuse reads the persisted rate at startup.

The default deployable policy is safety-constrained. Raw PPO is retained for
controlled comparisons; it is not a recommendation for an unvalidated external API.
The fixed in-flight ceiling and global/client enforcement remain in every mode.

## Optional prediction

The histogram gradient-boosted classifier estimates the risk of a latency/failure
threshold violation in any of the next three intervals. It uses 100 fitting
scenarios (seeds 1000–1099), separate sigmoid-calibration scenarios 30000–30019,
and test scenarios 50000–50019. Neighboring rows from a trajectory never cross
these groups. AUC, average precision, Brier and reliability curves are retained.

The target describes future outcomes under a sampled behavior policy. It is not
the causal result of a candidate budget and cannot anticipate every unannounced
backend failure. Good offline prediction is insufficient justification for a more
complex live controller. Simulator comparisons include prediction+AIMD, raw PPO,
PPO+prediction, PPO+safety and the combined hybrid. The predictor remains optional.

## Benchmarks

The original `rlrc calibrate`, `rlrc suite`, traffic generator, and raw reporting
remain available. ML comparisons reuse the same subprocess isolation, Redis
namespaces, admission scripts, calibration and request accounting.

```powershell
$env:RLRC_CONFIG = 'configs/portable.yaml'
$env:RLRC_REDIS_URL = 'redis://127.0.0.1:16379/0' # only for the local verification Redis
uv run --extra ml rlrc ml fidelity --output results/ml-runs/fidelity-new
uv run --extra ml rlrc ml sim --output results/ml-runs/simulator-new --repetitions 5 --calibration results/ml-runs/fidelity-new/calibration/calibration.json
uv run --extra ml rlrc ml real --output results/ml-runs/http-smoke-new --calibration results/ml-runs/fidelity-new/calibration/calibration.json --repetitions 1 --smoke
uv run --extra ml rlrc ml real --output results/ml-runs/http-new --calibration results/ml-runs/fidelity-new/calibration/calibration.json --repetitions 3
```

The real focused protocol is 33 trials: eleven configurations, three matched seeds,
30-second warm-up and 100-second measurement, randomized policy order. It includes
60 seconds after capacity restoration to give the original slow controller room
to recover. It takes roughly 72 minutes plus startup/report generation. Static
rates came from the independent real sweep, not from final-test tuning. A smoke
run is only an integration check, never evidence of a general improvement.

Longer optional workload definitions are in `configs/ml-scenarios.yaml`:

```powershell
uv run --extra ml rlrc ml real --scenario-file configs/ml-scenarios.yaml --scenario gradual --policies adaptive rl rl_safe --repetitions 3 --calibration results/ml-runs/fidelity-new/calibration/calibration.json --output results/ml-runs/gradual-new
```

The focused study is not a claim that this larger real workload matrix ran.
Multiple repetitions are independent workload seeds for fixed trained models;
retraining-seed uncertainty is not estimated. The generator rejects invalid late
launches; reports exclude runs with excessive misses or model fallback. All raw
evidence, source snapshots, failed experiments and checkpoints remain available.

Rebuild the dedicated report:

```powershell
uv run --extra ml python -m rlrc.ml.reporting --real results/ml-runs/real-capacity-v1 --sim results/ml-runs/heldout-simulator-final --fidelity results/ml-runs/fidelity-v1 --output results/rl-upgrade
```

It explicitly labels incomplete studies. Successful-response latency remains
separate from rejection latency. Goodput always means successful work within the
latency target, divided by measured workload duration. Simulator evaluation drains
pending replies and assigns measured results by request arrival cohort; training
uses delayed per-transition completions.

## Validation and next experiments

```powershell
$env:TEST_REDIS_URL = 'redis://127.0.0.1:16379/15'
$env:REQUIRE_REDIS = '1'
uv run --extra dev --extra ml pytest -q
uv run --extra dev --extra ml ruff check .
uv run --extra dev --extra ml ruff format --check .
uv run --extra dev --extra ml mypy --platform linux src
uv run --extra dev --extra ml mypy --platform win32 src
docker compose config --quiet
uv build
```

The original tests remain; new tests exercise the environment checker, determinism,
features, normalization, reward counterexamples, actions, safety, model loading,
checksum rejection, inference failure/recovery, real Redis publication, numerical
actor parity and checkpoint resume. CI installs the optional ML stack with CPU
Torch on Linux. A prior Windows-only subprocess typing issue was corrected without
changing its runtime behavior.

An additional conventional search (`python -m rlrc.ml.tuning`) selected faster AIMD
settings on validation only. `python -m rlrc.ml.tuning --heldout` tests that frozen
selection. This supplemental analysis preserves the original baseline and is
reported separately. `python -m rlrc.ml.audit` records post-training no-demand,
low-demand, heavy-overload and below-minimum-capacity cases without selecting models.

The edge audit exposes a demand-awareness limitation: raw PPO raises an idle budget
to the ceiling; safety blocks this with no observations, but both variants can grow
the unused budget with 5 RPS of healthy traffic. A backend below the minimum budget
can still overload. These are documented negative results, not silently fixed
against final evaluation data.

Priorities after the measured study are multiple training seeds, better
service/network calibration,
real independent upstreams, longer held-out HTTP families and paired uncertainty
estimates. Fixed quotas, a 10 RPS floor, incomplete observability and unforeseen
capacity changes remain limitations. The project makes no claim that adding RL
alone is novel or that one local study proves production superiority.
