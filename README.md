# Recoverability-aware Robot Manipulation Intervention

This repository is an executable research prototype derived from the Korean
research plan in this folder. It tests whether choosing among `continue`,
`recovery`, and `reset` using predicted task success and execution cost improves
the success/cost trade-off on ManiSkill's `PickCube-v1` task.

## What is implemented

- CPU-based ManiSkill state simulation (`physx_cpu`, privileged `state_dict`)
- A scripted nominal PickCube controller and an open/retreat/reacquire recovery
  controller using `pd_ee_delta_pos`
- Control-noise, action-dropout, and object-displacement failure perturbations
- Restorable simulator checkpoints and matched counterfactual rollouts from each
  checkpoint
- A PyTorch multi-head estimator conditioned on state, intervention option, and
  remaining budget
- Success-probability temperature calibration and ECE/Brier/NLL reporting
- A closed-loop success-minus-cost policy compared with no-intervention and
  risk-threshold recovery baselines, plus a success-only cost ablation
- Episode-grouped train/validation/test splits, so variants of the same
  checkpoint never appear in different splits

The pilot-dependent choices that the plan does not fix (perturbation range,
costs, thresholds, budgets, controller tolerances, and dataset size) are exposed
in [`configs/pick_cube.toml`](configs/pick_cube.toml).

Use [`configs/pick_cube_pilot.toml`](configs/pick_cube_pilot.toml) for a short
data-quality run before launching the larger core configuration. Long-running
commands now report collection, training, and evaluation progress on stderr.

After a healthy core run, execute validation-only tuning, independent seeded
runs, decision tracing, ablations, hierarchical confidence intervals, and plots
with:

```bash
python -m pip install -e ".[research]"
recoverability --config configs/pick_cube.toml research-suite \
  --dataset data/counterfactual.npz \
  --output-dir artifacts/research_suite \
  --seeds 7 17 29
```

## Setup

Python 3.11 is recommended. For the supplied CPU end-effector controller, use
WSL2/Linux: ManiSkill 3.0.1's CPU IK depends on Pinocchio/MPLib, and its pip
setup does not provide that dependency on native Windows. State-only experiments
do not require a display or Vulkan renderer. The environment adapter omits the
PickCube task's unused visual primitives during scene construction so state-only
simulation also works under WSL, where ManiSkill rendering is unsupported.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# WSL/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The implementation follows ManiSkill's documented `PickCube-v1`, privileged
`state_dict`, CPU wrapper, normalized EE controller, and simulator state restore
APIs:

- <https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html>
- <https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html>
- <https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html>
- <https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html#handling-custom-states>

## Run the experiment

First validate the environment and nominal controller:

```bash
recoverability --config configs/pick_cube.toml inspect
```

Then run each research stage separately:

```bash
recoverability --config configs/pick_cube.toml collect \
  --output data/counterfactual.npz

recoverability --config configs/pick_cube.toml train \
  --dataset data/counterfactual.npz \
  --model models/recoverability.pt \
  --metrics artifacts/training_metrics.json

recoverability --config configs/pick_cube.toml evaluate \
  --model models/recoverability.pt \
  --output-dir artifacts/evaluation
```

Or execute all three stages:

```bash
recoverability --config configs/pick_cube.toml run-all
```

Evaluation writes `summary.json`, paired per-episode results to `episodes.csv`,
decision-level predictions and choices to `decisions.csv`, and summarized failed
episodes to `failure_cases.json`. The report includes overall metrics, robustness
stratified by perturbation, and paired bootstrap confidence intervals versus each
baseline. The primary comparison fields are success rate, intervention frequency,
mean interventions, mean steps, and mean execution cost.

## Research protocol

1. A seeded nominal/perturbed trajectory is generated.
2. Periodic states and the states around a perturbation are checkpointed.
3. For every feasible `(checkpoint, option, remaining_budget)` combination, the
   simulator is restored and rolled forward. The label is final success; cost is
   simulated steps plus configured recovery/reset charges. Unused budget can
   fund recovery retries at the configured interval, so budget changes outcomes
   rather than serving only as a feasibility mask.
4. An MLP jointly predicts the success logit and log execution cost. Temperature
   scaling is fit only on the validation episodes.
5. During evaluation, the recoverability policy maximizes
   `P(success) - cost_weight * E(cost)` over budget-feasible options every few
   control steps.

The supplied configuration is deliberately a pilot configuration, not a claim
that its perturbation ranges or cost weights are scientifically optimal. Tune
them after reviewing nominal success and outcome balance. In particular, verify
that the dataset contains both successes and failures before interpreting
calibration metrics.

## Tests

The dependency-light tests cover configuration parsing, deterministic feature
flattening, group splitting, controller direction, and budget-constrained policy
selection:

```bash
python -m unittest discover -s tests -v
```

The simulator integration check is the `inspect` command and requires the full
project dependencies.
