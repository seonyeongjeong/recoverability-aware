# Recoverability-aware 프로젝트 설명

## 1. 한 문장 요약

이 프로젝트는 ManiSkill의 `PickCube-v1`에서 robot이 작업 도중 실패할 가능성을 예측하고, 매 decision point마다 `continue`, `recovery`, `reset` 중 하나를 골라 **성공률과 실행 비용의 trade-off를 개선할 수 있는지** 실험한 executable research prototype이다.

결론부터 말하면, pipeline의 구현 가능성과 일부 유용한 signal은 확인했지만 현재 결과만으로 learned `recoverability policy`가 단순한 baseline보다 안정적으로 우수하다고 말할 수는 없다. 단일 seed에서는 매우 좋은 결과가 나왔지만 independent multi-seed 실험에서 성능 변동이 컸다.

## 2. 연구 질문과 전체 구조

핵심 연구 질문은 다음과 같다.

> 현재 state와 남은 intervention budget이 주어졌을 때 각 intervention option의 최종 task success probability와 expected execution cost를 예측하고, 둘을 함께 고려해 행동을 고르면 단순한 intervention rule보다 나은가?

전체 pipeline은 다음 순서다.

1. ManiSkill `PickCube-v1`에서 scripted nominal controller를 실행한다.
2. episode에 failure perturbation을 넣고 중요한 simulator state를 checkpoint로 저장한다.
3. 같은 checkpoint에서 `continue`, `recovery`, `reset`을 각각 실행하는 matched counterfactual rollout을 만든다.
4. 각 rollout의 final success와 execution cost를 label로 저장한다.
5. state, option, remaining budget을 입력받아 success probability와 cost를 함께 예측하는 multi-head MLP를 학습한다.
6. closed-loop evaluation에서 일정 step마다 policy가 option을 다시 선택한다.
7. baseline, multi-seed, bootstrap confidence interval, ablation, perturbation별 robustness로 결과를 비교한다.

## 3. Simulation environment와 controller

- Environment: ManiSkill 3의 `PickCube-v1`
- Observation: `state_dict` 기반 privileged state
- Control mode: `pd_ee_delta_pos`
- Simulation: `physx_cpu`, headless state-only simulation
- Robot action: end-effector XYZ 이동 3차원 + gripper 1차원
- Core episode horizon: 최대 70 steps

`ScriptedPickCubeController`는 `APPROACH → DESCEND → CLOSE → LIFT → TRANSFER → SETTLE` phase로 cube를 집어 goal까지 옮긴다. nominal `continue`가 물체를 놓친 뒤 몰래 재시도하지 않도록 설계되어 있으며, 물체를 다시 잡는 동작은 별도 `recovery` option으로 분리되어 있다.

`RecoveryThenNominalController`는 gripper를 열고 물체 위로 retreat한 뒤 nominal controller를 새로 시작해 reacquire를 시도한다.

WSL/Linux에서 CPU end-effector controller를 실행하도록 되어 있다. state-only 실험에 불필요한 visual primitive 생성을 우회해 Vulkan renderer 없이 동작하도록 adapter가 들어 있다.

## 4. Data는 어떻게 얻었는가

외부에서 수집하거나 사람이 annotation한 data가 아니다. 모두 ManiSkill simulator 안에서 직접 생성한 **counterfactual rollout data**다.

### 4.1 Source trajectory와 perturbation

Core config에서는 60개 source episodes를 생성한다. 각 episode에는 90% 확률로 다음 중 하나의 perturbation을 넣는다.

| Perturbation | Sampling weight | 내용 |
|---|---:|---|
| `action_noise` | 0.25 | 12 steps 동안 XYZ action에 큰 Gaussian noise 추가 |
| `action_dropout` | 0.15 | 18 steps 동안 XYZ command를 사실상 0으로 만듦 |
| `gripper_release` | 0.25 | 4 steps 동안 gripper open command를 강제 |
| `object_shift` | 0.35 | cube를 수평 방향으로 0.10 m 이동 |

perturbation은 대체로 step 10~24 구간에 시작한다. episode마다 하나의 perturbation family만 선택한다.

### 4.2 Checkpoint 수집

step 6부터 6-step 간격으로 state를 저장하고, perturbation 시작/종료 주변 state도 추가한다. episode당 최대 8 checkpoints다. 저장되는 내용은 simulator `state_dict`, observation, episode/step ID, controller phase와 close progress다.

주의할 점은 dataset의 `perturbation` label이 해당 episode에 배정된 family를 뜻한다는 것이다. 그 episode의 perturbation 발생 전 checkpoint도 같은 family로 집계될 수 있으므로, perturbation별 dataset 통계가 오직 “이미 perturbation을 받은 state”만 분리한 것은 아니다.

### 4.3 Matched counterfactual rollout

각 checkpoint를 복원한 뒤 가능한 option을 모두 실행한다.

- Budget 0: `continue`
- Budget 1: `continue`, `recovery`
- Budget 2: `continue`, `recovery`, `reset`

각 조합을 2번씩 실행하며 작은 rollout action noise를 추가한다. `reset`은 완전히 다른 task가 아니라 같은 episode seed의 initial state로 돌아간다. 이 때문에 동일 goal/object configuration을 유지한 비교가 된다.

Label은 다음과 같다.

- `success`: rollout 종료 시 PickCube task 성공 여부
- `execution_cost`: 사용한 simulation steps + fixed intervention charge
- Fixed charge: `recovery = 8`, `reset = 20`, step당 cost = 1
- Resource budget: `recovery = 1`, `reset = 2`

Core dataset은 436 checkpoints × checkpoint당 12 samples = 5,232 samples다.

### 4.4 입력 feature

총 49개 state/controller features를 사용한다.

- robot `qpos` 9개와 `qvel` 9개
- `goal_pos`, `obj_pose`, `tcp_pose`
- object-to-goal, TCP-to-object 상대 위치
- `is_grasped`
- controller `phase` one-hot 6개와 `close_progress`

모델 입력에는 이 49개 feature 외에 intervention option one-hot 3개와 normalized remaining budget 1개가 붙는다.

### 4.5 Core dataset 품질

| 항목 | 값 |
|---|---:|
| Samples | 5,232 |
| Source episodes | 60 |
| Checkpoints | 436 |
| Successes / failures | 3,868 / 1,364 |
| Failure episodes | 58 |
| Overall success rate | 73.93% |
| Mean rollout cost | 31.56 |

Option별 counterfactual success rate는 `continue` 66.51%, `recovery` 72.02%, `reset` 100%다. 즉 recovery가 어려운 state 일부를 살리고, reset은 안정적이지만 fixed cost가 높은 의도한 구조가 data에 나타났다.

Perturbation family별 success rate는 `action_dropout` 97.89%, `action_noise` 91.16%, `gripper_release` 66.19%, `object_shift` 56.30%였다. 뒤의 두 family가 주된 hard case다.

초기 `pilot_v1` data는 2,184 samples 중 failure가 50개뿐이고 그것도 한 episode에 몰려 있어 validation/test에 negative class가 없는 문제가 있었다. Core run에서는 failure를 58 episodes에 분산시켜 이 문제를 보완했다.

## 5. 학습하는 model

`RecoverabilityNet`은 state + option + budget을 입력받는 shared MLP trunk와 두 개의 head로 구성된다.

- `success_head`: final success logit 예측
- `log_cost_head`: `log(1 + execution_cost)` 예측
- Core hidden layers: 128, 128 ReLU
- Loss: binary cross entropy + `cost_loss_weight × cost MSE`
- Optimizer: AdamW
- Core training: 80 epochs, batch size 256

같은 source episode에서 나온 여러 checkpoint와 option variant가 train/validation/test에 흩어지면 leakage가 생기므로 **episode-grouped split**을 사용한다. 각 split에 success와 failure가 모두 들어가도록 shuffle을 재시도한다.

Success probability에는 validation split만 사용한 temperature scaling을 적용한다. 평가 지표는 Accuracy, NLL, Brier score, ECE, cost MAE다.

Core model 결과는 다음과 같다.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 90.26% | 91.54% |
| Calibrated NLL | 0.1728 | 0.1783 |
| Calibrated Brier | 0.0584 | 0.0564 |
| Calibrated ECE | 0.0374 | 0.0478 |
| Cost MAE | 41.83 | 9.89 |

Temperature는 1.359였다. Success classification/calibration은 양호한 편이지만 validation과 test의 cost MAE 차이가 매우 커 cost head가 split과 seed에 불안정하다는 신호가 이미 보인다.

## 6. 어떤 policy를 실험했는가

평가 시 5 steps마다 decision을 내리며 initial budget은 2다.

### `recoverability`

Budget으로 가능한 option 각각에 대해 model prediction을 얻고 아래 utility가 가장 큰 option을 선택한다.

`utility(option) = P(success | state, option, budget) - λ × E(cost | state, option, budget)`

Core run의 λ는 0.015였고, validation-only tuning 후 research suite에서는 0.005를 사용했다.

### `success_only_ablation`

같은 learned estimator를 쓰지만 λ = 0으로 두고 predicted success만 최대화한다. Dataset에서 reset success가 항상 100%였기 때문에 불필요한 early reset을 고르는 경향이 강했다. Cost signal의 필요성을 보는 ablation이다.

### `no_intervention`

항상 `continue`를 선택한다.

### `threshold_recovery`

Predicted `continue` success가 threshold보다 낮으면, budget이 허용하는 동안 `recovery`를 선택하고 아니면 `continue`한다. `reset`은 선택하지 않는 단순 baseline이다. Core threshold는 0.45, validation tuning 후에는 0.75다.

모든 policy는 같은 episode ID, seed, perturbation schedule로 paired evaluation된다.

## 7. 결과

### 7.1 단일 Core run

각 policy를 같은 50 episodes에서 평가했다.

| Policy | Success | Mean interventions | Mean cost |
|---|---:|---:|---:|
| `recoverability` | 92% | 0.50 | 57.44 |
| `success_only_ablation` | 42% | 1.00 | 77.54 |
| `no_intervention` | 42% | 0.00 | 57.54 |
| `threshold_recovery` | 94% | 0.66 | 54.68 |

이 run만 보면 learned policy는 no intervention보다 success가 50 percentage points 높고 평균 cost 차이는 -0.10으로 거의 없다. 하지만 threshold recovery보다 success가 2 points 낮고 cost도 2.76 높았다. 따라서 좋은 단일 run에서도 learned policy가 가장 강한 baseline을 이기지는 못했다.

### 7.2 Validation-only tuning

Research suite는 test/closed-loop 결과를 보지 않고 validation counterfactual만 사용해 다음을 선택했다.

- `cost_loss_weight`: 후보 0.05, 0.10, 0.25, 0.50 중 **0.50**
- Decision λ: 후보 0, 0.005, 0.01, 0.015, 0.02, 0.03 중 **0.005**
- Threshold: 후보 0.25~0.75 중 **0.75**

Cost-head tuning score는 `validation NLL + 0.25 × validation cost MAE / validation mean cost`다. Decision rule은 validation counterfactual에서 `success rate - 0.015 × mean cost`가 최대가 되도록 골랐다.

### 7.3 Multi-seed 결과가 최종적으로 더 중요하다

Seeds 7, 17, 29에서 각각 새 collection/training을 수행하고, seed마다 50 paired evaluation episodes를 실행했다. Seed 7은 기존 core dataset을 재사용하고 seeds 17/29는 새 dataset을 수집했다. Hierarchical bootstrap은 먼저 seed를, 그 안에서 episode를 resample한다.

| Policy | Mean success (95% CI) | Mean cost (95% CI) |
|---|---:|---:|
| `recoverability` | 54.7% (32.7%, 87.3%) | 67.23 (59.47, 78.46) |
| `success_only_ablation` | 38.7% (30.7%, 46.7%) | 77.65 (74.76, 80.39) |
| `no_intervention` | 38.7% (30.7%, 46.7%) | 57.67 (54.79, 60.43) |
| `threshold_recovery` | 61.3% (36.7%, 95.3%) | 67.41 (55.89, 75.29) |

`recoverability`는 no intervention보다 평균 success가 +16.0 points였지만 95% CI가 -1.3~+43.3 points로 zero를 포함한다. Cost는 +9.55이고 CI 2.54~20.00으로 더 높았다. Threshold recovery와 비교하면 success -6.7 points, cost -0.18이며 두 CI 모두 zero를 포함한다.

즉 **현재 multi-seed evidence로는 recoverability가 no intervention 또는 threshold recovery보다 우수하다는 결론을 낼 수 없다.**

Seed별 learned policy success는 88%, 36%, 40%였다. Seed 17에서는 모든 episode에서 reset을 한 번씩 골라 cost만 20 늘고 no-intervention success 36%를 전혀 개선하지 못했다. Seed 29에서는 intervention이 전체 50 episodes에서 16회뿐이었고, 실패한 30 episodes 중 27개에는 intervention이 전혀 없었다. 서로 반대 방향의 policy collapse가 나타난 것이다.

### 7.4 Perturbation별 결과

Multi-seed 평균에서 no intervention도 `action_dropout` 100%, `action_noise` 91.2%, nominal 100%로 이미 잘했다. 어려운 family는 다음과 같다.

| Perturbation | No intervention | Recoverability | Threshold recovery |
|---|---:|---:|---:|
| `gripper_release` | 2.6% | 21.1% | 34.2% |
| `object_shift` | 1.9% | 34.0% | 43.4% |

Learned intervention이 hard case를 일부 회복시키기는 했지만 threshold baseline이 더 높았다.

### 7.5 Ablation

Ablation은 seed 7 한 개에서만 실행했으므로 diagnostic 결과이지 robust causal estimate는 아니다.

| Ablation | Training samples | Success | Mean cost |
|---|---:|---:|---:|
| No controller-phase features | 5,232 | 44% | 76.34 |
| No cost head/decision cost | 5,232 | 42% | 77.54 |
| Without action noise | 4,248 | 46% | 76.28 |
| Without action dropout | 4,284 | 62% | 69.90 |
| Without gripper release | 4,392 | 90% | 59.26 |
| Without object shift | 3,024 | 50% | 72.96 |

Favorable seed 7에서는 controller phase와 cost signal을 제거하면 성능이 no-intervention 수준으로 떨어졌다. `object_shift` data를 빼도 크게 악화되어 이 hard case의 training data가 중요해 보인다. 반면 `gripper_release`를 제거했을 때 오히려 90%로 높아진 것은 dataset balance 또는 spurious correlation 가능성을 경고하며, 한 seed만으로 긍정적으로 해석하면 안 된다.

## 8. 결과 해석과 한계

현재 evidence가 지지하는 내용은 다음과 같다.

- Restorable checkpoint에서 matched counterfactual data를 만들고 success/cost estimator를 학습하는 pipeline은 작동한다.
- Recovery는 continue보다 counterfactual success가 높고, controller phase와 cost signal은 적어도 한 seed에서 decision에 유용했다.
- `gripper_release`, `object_shift` 같은 hard failure에는 intervention이 실제 도움을 줄 수 있다.

아직 지지하지 못하는 내용은 다음과 같다.

- Learned policy가 단순 threshold recovery보다 우수하다는 주장
- Independent seed에서도 안정적으로 no intervention을 이긴다는 주장
- 현재 cost head가 일반화되고 안정적으로 calibrated되어 있다는 주장

추가로 주의할 구현/설정 사항이 있다.

- Code는 남은 budget으로 periodic fallback recovery를 추가하는 기능을 지원하지만 현재 core config의 `fallback_recovery_interval = 0`은 이를 끈다. 따라서 현재 run에서 remaining budget은 주로 feasible option을 제한하고 model input으로 쓰이며, 같은 option의 후속 retry 횟수를 바꾸지는 않는다. README의 “budget changes outcomes through retries” 설명은 현재 config에는 그대로 적용되지 않는다.
- `reset`은 같은 episode seed의 initial state로 돌아가므로 결정론적 실패 원인이 남으면 성공을 바꾸지 못하고 cost만 늘릴 수 있다. Seed 17이 대표 사례다.
- Collection seed와 model training seed가 run 단위로 함께 바뀐다. 현재 설계로는 dataset variation과 optimization variation의 영향을 분리하기 어렵다.
- Reset rollout은 core data에서 100% 성공해 success-only policy가 reset으로 쏠리기 쉽다.
- Dataset 및 evaluation이 privileged state와 한 task, 한 scripted controller에 한정되어 있어 visual observation이나 다른 manipulation task로 일반화되었다고 볼 수 없다.
- Multi-seed가 3개뿐이라 hierarchical CI가 매우 넓다.

다음 우선순위는 cost target normalization/bounded output 등으로 cost head를 안정화하고, deterministic reset loop 방지 rule을 넣고, collection seed와 training seed를 분리한 뒤 더 많은 independent seeds에서 반복하는 것이다. Ablation도 multi-seed로 확장한 후 다른 task나 visual observation으로 넘어가는 편이 타당하다.

## 9. 폴더별 안내

- `README.md`: setup, command, research protocol 개요
- `configs/pick_cube.toml`: core experiment 설정
- `configs/pick_cube_pilot.toml`: 짧은 pilot 설정
- `src/recoverability/collection.py`: checkpoint와 counterfactual data 생성
- `src/recoverability/controllers.py`: nominal/recovery controller
- `src/recoverability/perturbations.py`: 네 가지 failure perturbation
- `src/recoverability/model.py`: multi-head estimator, training, calibration
- `src/recoverability/policies.py`: 네 policy의 선택 규칙
- `src/recoverability/evaluation.py`: paired closed-loop evaluation, bootstrap용 episode/decision 기록
- `src/recoverability/research.py`: tuning, multi-seed, ablation, hierarchical bootstrap, plot 생성
- `data/counterfactual.npz`: core counterfactual dataset
- `models/recoverability.pt`: core trained model
- `artifacts/evaluation/`: 단일 core evaluation 결과
- `artifacts/core_report.md`: core와 multi-seed를 함께 해석한 상세 영문 report
- `artifacts/pilot_v1/`: failure가 지나치게 적었던 초기 pilot
- `artifacts/pilot_v3/`: perturbation/controller를 조정한 small pilot
- `artifacts/research_suite/seeds/`: seed별 dataset, model, metrics, episode/decision/failure trace
- `artifacts/research_suite/ablations/`: ablation별 model과 evaluation
- `artifacts/research_suite/plots/`: multi-seed, robustness, calibration, ablation plot
- `artifacts/research_suite/aggregate.json`: 최종 hierarchical bootstrap 수치
- `docs/IMPLEMENTATION_NOTES.md`: 원 연구계획의 요구사항을 code로 어떻게 operationalize했는지 설명
- `tests/`: config, data split, controller, perturbation, policy, research utility에 대한 dependency-light tests
- 연구수행계획서 PDF: 이 prototype의 출발점이 된 Korean research plan

현재 code 기준 unit tests는 19개 모두 통과한다. 다만 simulator integration 자체는 unit test가 아니라 ManiSkill이 설치된 WSL/Linux 환경에서 `recoverability --config configs/pick_cube.toml inspect`로 별도 확인해야 한다.
