from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np


# Ubuntu 22.04 packages Python 3.11.0rc1, which identifies as Python 3.11 but
# predates these final-3.11 APIs. Torch Dynamo imports them while AdamW is being
# initialized. This project does not parse untrusted integer strings, so the
# prerelease interpreter's original unlimited behavior is the appropriate
# compatibility fallback until the environment is upgraded to stable Python.
if sys.version_info >= (3, 11) and not hasattr(sys, "get_int_max_str_digits"):
    def _get_int_max_str_digits() -> int:
        return 0

    def _set_int_max_str_digits(maxdigits: int) -> None:
        del maxdigits

    sys.get_int_max_str_digits = _get_int_max_str_digits  # type: ignore[attr-defined]
    sys.set_int_max_str_digits = _set_int_max_str_digits  # type: ignore[attr-defined]

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised only without ML dependencies
    raise RuntimeError(
        "PyTorch is required for estimator training/inference. Install with `pip install -e .`."
    ) from exc

from .config import TrainingConfig
from .dataset import CounterfactualDataset
from .domain import InterventionOption, Prediction


MODEL_SCHEMA_VERSION = 1
OPTION_COUNT = len(InterventionOption)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecoverabilityNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.success_head = nn.Linear(previous, 1)
        self.log_cost_head = nn.Linear(previous, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(inputs)
        return self.success_head(hidden).squeeze(-1), self.log_cost_head(hidden).squeeze(-1)


def _normalizer(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = states.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _build_inputs(
    states: np.ndarray,
    option_ids: np.ndarray,
    budgets: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    max_budget: float,
) -> np.ndarray:
    normalized_states = (states.astype(np.float32) - state_mean) / state_std
    options = np.zeros((len(states), OPTION_COUNT), dtype=np.float32)
    options[np.arange(len(states)), option_ids.astype(np.int64)] = 1.0
    budget_scale = max(max_budget, 1.0)
    normalized_budgets = budgets.astype(np.float32).reshape(-1, 1) / budget_scale
    return np.concatenate([normalized_states, options, normalized_budgets], axis=1)


def _dataset_tensors(
    dataset: CounterfactualDataset,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    max_budget: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = _build_inputs(
        dataset.states,
        dataset.option_ids,
        dataset.remaining_budgets,
        state_mean,
        state_std,
        max_budget,
    )
    return (
        torch.from_numpy(inputs),
        torch.from_numpy(dataset.successes.astype(np.float32)),
        torch.from_numpy(np.log1p(dataset.costs).astype(np.float32)),
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= boundaries[index]) & (
                probabilities <= boundaries[index + 1]
            )
        else:
            mask = (probabilities >= boundaries[index]) & (
                probabilities < boundaries[index + 1]
            )
        if mask.any():
            confidence = float(probabilities[mask].mean())
            accuracy = float(labels[mask].mean())
            result += float(mask.mean()) * abs(confidence - accuracy)
    return result


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if len(logits) == 0 or torch.unique(labels).numel() < 2:
        return 1.0
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=60)
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu())


def _metrics(
    model: RecoverabilityNet,
    dataset: CounterfactualDataset,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    max_budget: float,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    inputs, labels, log_cost_targets = _dataset_tensors(
        dataset, state_mean, state_std, max_budget
    )
    model.eval()
    with torch.no_grad():
        logits, log_costs = model(inputs.to(device))
        probabilities = torch.sigmoid(logits / temperature).cpu().numpy()
        predicted_costs = torch.expm1(log_costs.clamp(min=0.0)).cpu().numpy()
    labels_np = labels.numpy()
    eps = 1e-7
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    nll = -np.mean(labels_np * np.log(clipped) + (1.0 - labels_np) * np.log(1.0 - clipped))
    return {
        "nll": float(nll),
        "brier": float(np.mean((probabilities - labels_np) ** 2)),
        "ece": expected_calibration_error(probabilities, labels_np),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels_np)),
        "cost_mae": float(np.mean(np.abs(predicted_costs - np.expm1(log_cost_targets.numpy())))),
    }


def train_estimator(
    dataset: CounterfactualDataset,
    config: TrainingConfig,
    output_path: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    train, validation, test = dataset.split_by_episode(
        config.validation_fraction,
        config.test_fraction,
        config.seed,
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _device()
    state_mean, state_std = _normalizer(train.states)
    max_budget = float(max(dataset.remaining_budgets.max(initial=1.0), 1.0))
    train_tensors = _dataset_tensors(train, state_mean, state_std, max_budget)
    validation_tensors = _dataset_tensors(validation, state_mean, state_std, max_budget)
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=min(config.batch_size, len(train)),
        shuffle=True,
    )
    model = RecoverabilityNet(train_tensors[0].shape[1], config.hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    success_loss = nn.BCEWithLogitsLoss()
    cost_loss = nn.MSELoss()
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    progress_interval = max(1, config.epochs // 10)
    for epoch in range(config.epochs):
        model.train()
        for inputs, labels, log_cost_targets in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            log_cost_targets = log_cost_targets.to(device)
            logits, log_costs = model(inputs)
            loss = success_loss(logits, labels) + config.cost_loss_weight * cost_loss(
                log_costs, log_cost_targets
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_inputs, val_labels, val_log_costs = (
                tensor.to(device) for tensor in validation_tensors
            )
            val_logits, val_cost_predictions = model(val_inputs)
            validation_loss = success_loss(val_logits, val_labels) + config.cost_loss_weight * cost_loss(
                val_cost_predictions, val_log_costs
            )
        if float(validation_loss) < best_validation:
            best_validation = float(validation_loss)
            best_state = copy.deepcopy(model.state_dict())
        if progress is not None and (
            epoch == 0 or (epoch + 1) % progress_interval == 0 or epoch + 1 == config.epochs
        ):
            progress(
                f"training epoch {epoch + 1}/{config.epochs}: "
                f"validation_loss={float(validation_loss):.6f}"
            )

    if best_state is None:  # pragma: no cover - positive epochs guaranteed by normal config
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_inputs, val_labels, _ = (
            tensor.to(device) for tensor in validation_tensors
        )
        validation_logits, _ = model(val_inputs)
    temperature = _fit_temperature(validation_logits, val_labels)

    metrics: dict[str, Any] = {
        "device": str(device),
        "temperature": temperature,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "validation_uncalibrated": _metrics(
            model, validation, state_mean, state_std, max_budget, 1.0, device
        ),
        "validation_calibrated": _metrics(
            model, validation, state_mean, state_std, max_budget, temperature, device
        ),
        "test_uncalibrated": _metrics(
            model, test, state_mean, state_std, max_budget, 1.0, device
        ),
        "test_calibrated": _metrics(
            model, test, state_mean, state_std, max_budget, temperature, device
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "state_dict": model.state_dict(),
            "state_dim": dataset.states.shape[1],
            "feature_names": dataset.feature_names,
            "hidden_sizes": config.hidden_sizes,
            "state_mean": state_mean,
            "state_std": state_std,
            "max_budget": max_budget,
            "temperature": temperature,
            "training_config": asdict(config),
        },
        output,
    )
    return metrics


class RecoverabilityEstimator:
    def __init__(self, artifact: dict[str, Any], device: torch.device | None = None):
        if artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError("Unsupported estimator artifact schema.")
        self.device = device or _device()
        self.feature_names = tuple(artifact["feature_names"])
        self.state_mean = np.asarray(artifact["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(artifact["state_std"], dtype=np.float32)
        self.max_budget = float(artifact["max_budget"])
        self.temperature = float(artifact["temperature"])
        # Network input also contains one-hot option and scalar budget.
        self.model = RecoverabilityNet(
            int(artifact["state_dim"]) + OPTION_COUNT + 1,
            tuple(artifact["hidden_sizes"]),
        ).to(self.device)
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device | None = None,
    ) -> "RecoverabilityEstimator":
        selected = torch.device(device) if device is not None else _device()
        artifact = torch.load(Path(path), map_location=selected, weights_only=False)
        return cls(artifact, selected)

    def predict_many(
        self,
        states: np.ndarray,
        option_ids: np.ndarray,
        budgets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if states.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} state features, got {states.shape[1]}."
            )
        inputs = _build_inputs(
            states,
            np.asarray(option_ids),
            np.asarray(budgets),
            self.state_mean,
            self.state_std,
            self.max_budget,
        )
        with torch.no_grad():
            logits, log_costs = self.model(torch.from_numpy(inputs).to(self.device))
            probabilities = torch.sigmoid(logits / self.temperature)
            costs = torch.expm1(log_costs.clamp(min=0.0))
        return probabilities.cpu().numpy(), costs.cpu().numpy()

    def predict_options(
        self,
        state: np.ndarray,
        budget: int,
        options: tuple[InterventionOption, ...],
    ) -> dict[InterventionOption, Prediction]:
        states = np.repeat(np.asarray(state, dtype=np.float32)[None, :], len(options), axis=0)
        option_ids = np.asarray([int(option) for option in options], dtype=np.int64)
        budgets = np.full(len(options), budget, dtype=np.float32)
        probabilities, costs = self.predict_many(states, option_ids, budgets)
        return {
            option: Prediction(float(probability), float(cost))
            for option, probability, cost in zip(options, probabilities, costs, strict=True)
        }


class FeatureSubsetEstimator:
    """Adapt a model trained on a named feature subset to full simulator states."""

    def __init__(
        self,
        estimator: RecoverabilityEstimator,
        source_feature_names: tuple[str, ...],
    ):
        positions = {name: index for index, name in enumerate(source_feature_names)}
        missing = [name for name in estimator.feature_names if name not in positions]
        if missing:
            raise ValueError(f"Source state is missing feature(s): {', '.join(missing)}")
        self.estimator = estimator
        self.indices = np.asarray(
            [positions[name] for name in estimator.feature_names],
            dtype=np.int64,
        )

    def predict_options(
        self,
        state: np.ndarray,
        budget: int,
        options: tuple[InterventionOption, ...],
    ) -> dict[InterventionOption, Prediction]:
        selected = np.asarray(state, dtype=np.float32)[self.indices]
        return self.estimator.predict_options(selected, budget, options)
