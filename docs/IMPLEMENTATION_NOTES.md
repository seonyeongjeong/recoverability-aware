# Research-plan mapping and assumptions

The source PDF is a research execution plan rather than a detailed algorithm or
software interface. This file records how each requirement was operationalized
so later pilot work can change assumptions without silently changing the
research question.

| Plan item | Implementation |
| --- | --- |
| ManiSkill3, simple pick-and-place task | `PickCube-v1`, one CPU environment |
| Privileged state | `state_dict` observation, deterministically flattened for the estimator |
| Nominal/recovery controller scripted or motion-planned | Scripted EE-delta controllers in `controllers.py` |
| Control/object/environment-condition perturbations | Control noise/dropout and object displacement; new perturbations can implement the same schedule hook |
| Save important trajectory states | `BaseEnv.get_state_dict()` plus observation and step metadata |
| Run continue/recovery/reset from the same state | Each option restores a deep copy of the checkpoint; reset restores the episode's initial checkpoint |
| Estimate success probability and expected cost | Shared MLP trunk with success-logit and log-cost heads |
| Condition on state, recovery ID, remaining budget | Flat state + option one-hot + normalized budget |
| Evaluate calibration and optionally post-hoc calibrate | Brier score, NLL, ECE, validation-only temperature scaling |
| Closed-loop intervention | Budget-feasible utility maximization at a configurable interval |
| Baselines | No intervention and threshold-triggered recovery |
| Metrics | Success rate, intervention frequency/count, simulated steps, execution cost |
| Ablation/robustness/error analysis | Data and episode-level outputs are saved; task/condition extensions remain future experiment configuration |

## Explicit assumptions

- Budget is a discrete intervention resource. Recovery consumes one unit and
  reset consumes two by default. Continue consumes none.
- Execution cost is control steps plus a fixed recovery/reset charge. Budget
  and execution cost are separate concepts even though both penalize
  intervention.
- `reset` returns to the initial state of the same seeded episode, preserving a
  fair goal/object configuration.
- `pd_ee_delta_pos` preserves the initial end-effector orientation and uses four
  normalized action dimensions (XYZ plus gripper). The configuration assumes
  the Panda default mapping of a magnitude-one translation command to 0.1 m.
- The default CPU EE controller must be run in WSL2/Linux. ManiSkill 3.0.1 does
  not install its CPU Pinocchio/MPLib IK dependency on native Windows; a clear
  startup error is raised there instead of the simulator's opaque `NoneType`
  controller failure.
- Counterfactual rollout action jitter provides repeated stochastic trials from
  a checkpoint. It is configured independently from failure-inducing noise.
- A selected option is evaluated as a one-step decision followed by a simple
  downstream fallback: if the task is still running at each configured retry
  interval, unused budget buys another recovery attempt. This makes remaining
  budget causally affect both success and cost labels and gives the estimator a
  Q-value-like meaning.
- The current state checkpoint is sufficient for the non-target EE controller.
  If a stateful controller is substituted, its internal target state must also
  be added to `Checkpoint`.

## Pilot gates before a full run

1. `inspect` should show reliable nominal success across several seeds.
2. Perturbations should produce both recoverable and unrecoverable checkpoints;
   inspect per-option label balance before training.
3. Increase collection episodes only after confirming checkpoint restoration is
   deterministic on the target ManiSkill/SAPIEN version.
4. Run multiple collection/training/evaluation seeds for the reported study and
   include confidence intervals; the default single seed is for pipeline
   validation.
