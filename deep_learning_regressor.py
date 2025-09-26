"""Treina um regressor profundo (MLP) para aproximar características quânticas."""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Dataset:
    """Container for the regression matrices."""

    features: List[List[float]]
    targets: List[List[float]]
    feature_names: Sequence[str]
    target_names: Sequence[str]


class DeepMLPRegressor:
    """Minimal multi-layer perceptron regressor implemented without NumPy."""

    def __init__(
        self,
        hidden_layers: Sequence[int],
        learning_rate: float,
        epochs: int,
        batch_size: int,
        random_state: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self._weights: List[List[List[float]]] = []
        self._biases: List[List[float]] = []
        self._m_w: List[List[List[float]]] = []
        self._v_w: List[List[List[float]]] = []
        self._m_b: List[List[float]] = []
        self._v_b: List[List[float]] = []
        self._step = 0

    @staticmethod
    def _zeros_matrix(rows: int, cols: int) -> List[List[float]]:
        return [[0.0 for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def _zeros_vector(size: int) -> List[float]:
        return [0.0 for _ in range(size)]

    def _initialize(self, n_features: int, n_outputs: int) -> None:
        layer_sizes = [n_features, *self.hidden_layers, n_outputs]
        rng = random.Random(self.random_state)

        self._weights = []
        self._biases = []
        self._m_w = []
        self._v_w = []
        self._m_b = []
        self._v_b = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            weight_matrix = []
            for _ in range(in_dim):
                weight_matrix.append([rng.uniform(-limit, limit) for _ in range(out_dim)])
            bias_vector = [0.0 for _ in range(out_dim)]

            self._weights.append(weight_matrix)
            self._biases.append(bias_vector)
            self._m_w.append(self._zeros_matrix(in_dim, out_dim))
            self._v_w.append(self._zeros_matrix(in_dim, out_dim))
            self._m_b.append(self._zeros_vector(out_dim))
            self._v_b.append(self._zeros_vector(out_dim))

        self._step = 0

    @staticmethod
    def _relu(values: List[float]) -> List[float]:
        return [value if value > 0.0 else 0.0 for value in values]

    @staticmethod
    def _relu_grad(values: List[float]) -> List[float]:
        return [1.0 if value > 0.0 else 0.0 for value in values]

    @staticmethod
    def _matvec(weights: List[List[float]], vector: List[float]) -> List[float]:
        output_dim = len(weights[0])
        result = [0.0 for _ in range(output_dim)]
        for input_idx, weight_row in enumerate(weights):
            contribution = vector[input_idx]
            for output_idx in range(output_dim):
                result[output_idx] += weight_row[output_idx] * contribution
        return result

    @staticmethod
    def _add_vectors(vec1: List[float], vec2: List[float]) -> List[float]:
        return [a + b for a, b in zip(vec1, vec2)]

    def _forward_single(self, vector: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        activations = [vector]
        pre_activations: List[List[float]] = []
        current = vector

        for layer_idx, (weight, bias) in enumerate(zip(self._weights, self._biases)):
            linear = self._add_vectors(self._matvec(weight, current), bias)
            pre_activations.append(linear)
            if layer_idx == len(self._weights) - 1:
                current = linear
            else:
                current = self._relu(linear)
            activations.append(current)

        return pre_activations, activations

    def _backpropagate(
        self,
        expected: List[float],
        pre_acts: List[List[float]],
        acts: List[List[float]],
        grads_w: List[List[List[float]]],
        grads_b: List[List[float]],
    ) -> None:
        delta = [acts[-1][i] - expected[i] for i in range(len(expected))]

        for layer_idx in reversed(range(len(self._weights))):
            # Accumulate gradients for the current layer
            for input_idx, activation in enumerate(acts[layer_idx]):
                for output_idx, delta_value in enumerate(delta):
                    grads_w[layer_idx][input_idx][output_idx] += activation * delta_value
            for output_idx, delta_value in enumerate(delta):
                grads_b[layer_idx][output_idx] += delta_value

            if layer_idx != 0:
                next_delta = [0.0 for _ in range(len(self._weights[layer_idx - 1][0]))]
                relu_grad = self._relu_grad(pre_acts[layer_idx - 1])
                for input_idx in range(len(next_delta)):
                    error = 0.0
                    for output_idx, delta_value in enumerate(delta):
                        error += self._weights[layer_idx][input_idx][output_idx] * delta_value
                    next_delta[input_idx] = error * relu_grad[input_idx]
                delta = next_delta
            else:
                delta = []

    def _apply_adam(self, grads_w: List[List[List[float]]], grads_b: List[List[float]]) -> None:
        self._step += 1
        lr = self.learning_rate

        for layer_idx in range(len(self._weights)):
            for input_idx in range(len(self._weights[layer_idx])):
                for output_idx in range(len(self._weights[layer_idx][input_idx])):
                    grad = grads_w[layer_idx][input_idx][output_idx]
                    self._m_w[layer_idx][input_idx][output_idx] = (
                        self.beta1 * self._m_w[layer_idx][input_idx][output_idx]
                        + (1 - self.beta1) * grad
                    )
                    self._v_w[layer_idx][input_idx][output_idx] = (
                        self.beta2 * self._v_w[layer_idx][input_idx][output_idx]
                        + (1 - self.beta2) * grad * grad
                    )
                    m_hat = self._m_w[layer_idx][input_idx][output_idx] / (
                        1 - self.beta1 ** self._step
                    )
                    v_hat = self._v_w[layer_idx][input_idx][output_idx] / (
                        1 - self.beta2 ** self._step
                    )
                    self._weights[layer_idx][input_idx][output_idx] -= (
                        lr * m_hat / (math.sqrt(v_hat) + self.eps)
                    )

            for output_idx in range(len(self._biases[layer_idx])):
                grad_b = grads_b[layer_idx][output_idx]
                self._m_b[layer_idx][output_idx] = (
                    self.beta1 * self._m_b[layer_idx][output_idx] + (1 - self.beta1) * grad_b
                )
                self._v_b[layer_idx][output_idx] = (
                    self.beta2 * self._v_b[layer_idx][output_idx] + (1 - self.beta2) * grad_b * grad_b
                )
                m_hat_b = self._m_b[layer_idx][output_idx] / (1 - self.beta1 ** self._step)
                v_hat_b = self._v_b[layer_idx][output_idx] / (1 - self.beta2 ** self._step)
                self._biases[layer_idx][output_idx] -= lr * m_hat_b / (math.sqrt(v_hat_b) + self.eps)

    def fit(self, X: List[List[float]], y: List[List[float]]) -> None:
        if not X or not y:
            raise ValueError("Os dados de treinamento não podem estar vazios.")
        if len(X) != len(y):
            raise ValueError("O número de amostras e de alvos deve ser igual.")

        self._initialize(len(X[0]), len(y[0]))
        rng = random.Random(self.random_state)

        for _ in range(self.epochs):
            indices = list(range(len(X)))
            rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                self._current_batch_size = len(batch_indices)

                grads_w = [self._zeros_matrix(len(layer), len(layer[0])) for layer in self._weights]
                grads_b = [self._zeros_vector(len(bias)) for bias in self._biases]

                for idx in batch_indices:
                    pre_acts, acts = self._forward_single(X[idx])
                    self._backpropagate(y[idx], pre_acts, acts, grads_w, grads_b)

                batch_scale = 1.0 / max(1, len(batch_indices))
                for layer_idx in range(len(grads_w)):
                    for input_idx in range(len(grads_w[layer_idx])):
                        for output_idx in range(len(grads_w[layer_idx][input_idx])):
                            grads_w[layer_idx][input_idx][output_idx] *= batch_scale
                    for output_idx in range(len(grads_b[layer_idx])):
                        grads_b[layer_idx][output_idx] *= batch_scale

                self._apply_adam(grads_w, grads_b)

    def predict(self, X: List[List[float]]) -> List[List[float]]:
        if not self._weights:
            raise RuntimeError("O modelo ainda não foi treinado.")

        predictions: List[List[float]] = []
        for vector in X:
            _, activations = self._forward_single(vector)
            predictions.append(list(activations[-1]))
        return predictions


def read_feature_folds(data_dir: Path) -> Dataset:
    """Load all feature folds available in ``data_dir`` without NumPy or pandas."""

    fold_paths = sorted(data_dir.glob("features_y_fold*.csv"))
    if not fold_paths:
        raise FileNotFoundError(
            "Nenhum arquivo `features_y_fold*.csv` foi encontrado na pasta especificada."
        )

    feature_names: List[str] | None = None
    target_names: List[str] | None = None
    features: List[List[float]] = []
    targets: List[List[float]] = []

    for path in fold_paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if feature_names is None or target_names is None:
                columns = reader.fieldnames or []
                feature_names = [col for col in columns if col.startswith("class_")]
                target_names = [col for col in columns if col.startswith("qf_")]
                if not feature_names or not target_names:
                    raise ValueError(
                        "Não foi possível localizar as colunas de atributos clássicos ou alvos quânticos."
                    )

            for row in reader:
                features.append([float(row[name]) for name in feature_names])
                targets.append([float(row[name]) for name in target_names])

    return Dataset(
        features=features,
        targets=targets,
        feature_names=feature_names or [],
        target_names=target_names or [],
    )


def train_test_split(
    X: List[List[float]],
    y: List[List[float]],
    test_size: float,
    random_state: int,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size deve estar entre 0 e 1.")

    rng = random.Random(random_state)
    indices = list(range(len(X)))
    rng.shuffle(indices)

    test_count = max(1, int(round(len(indices) * test_size)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    if not train_indices:
        raise ValueError("Não há dados suficientes para o conjunto de treinamento.")

    X_train = [X[idx] for idx in train_indices]
    X_test = [X[idx] for idx in test_indices]
    y_train = [y[idx] for idx in train_indices]
    y_test = [y[idx] for idx in test_indices]

    return X_train, X_test, y_train, y_test


def evaluate_predictions(y_true: List[List[float]], y_pred: List[List[float]]) -> dict[str, float]:
    n_samples = len(y_true)
    n_targets = len(y_true[0]) if y_true else 0

    mse_sum = 0.0
    mae_sum = 0.0
    target_means = [0.0 for _ in range(n_targets)]

    for sample in y_true:
        for idx, value in enumerate(sample):
            target_means[idx] += value
    target_means = [value / n_samples for value in target_means]

    ss_tot = [0.0 for _ in range(n_targets)]
    ss_res = [0.0 for _ in range(n_targets)]

    for true_sample, pred_sample in zip(y_true, y_pred):
        for idx, (true_value, pred_value) in enumerate(zip(true_sample, pred_sample)):
            diff = true_value - pred_value
            mse_sum += diff * diff
            mae_sum += abs(diff)
            ss_res[idx] += diff * diff
            centered = true_value - target_means[idx]
            ss_tot[idx] += centered * centered

    mse = mse_sum / (n_samples * n_targets)
    rmse = math.sqrt(mse)
    mae = mae_sum / (n_samples * n_targets)

    r2_values = []
    for idx in range(n_targets):
        if ss_tot[idx] == 0.0:
            r2_values.append(0.0)
        else:
            r2_values.append(1 - ss_res[idx] / ss_tot[idx])
    r2 = sum(r2_values) / n_targets if r2_values else 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2}


def parse_hidden_layers(raw: Sequence[str]) -> Sequence[int]:
    try:
        layers = tuple(int(value) for value in raw)
    except ValueError as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError("Camadas ocultas devem ser números inteiros.") from exc

    if not layers:
        raise argparse.ArgumentTypeError("É necessário informar pelo menos uma camada oculta.")

    if any(size <= 0 for size in layers):
        raise argparse.ArgumentTypeError("Os tamanhos das camadas devem ser positivos.")

    return layers


def run(
    data_dir: Path,
    hidden_layers: Sequence[int],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    test_size: float,
    random_state: int,
    output_predictions: Path | None,
) -> dict[str, float]:
    dataset = read_feature_folds(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.targets,
        test_size=test_size,
        random_state=random_state,
    )

    model = DeepMLPRegressor(
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    if output_predictions is not None:
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        with output_predictions.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"pred_{name}" for name in dataset.target_names])
            for row in y_pred:
                writer.writerow(row)

    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("features"),
        help="Pasta contendo os arquivos features_y_fold*.csv.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=parse_hidden_layers,
        nargs="+",
        default=(128, 64, 32),
        help="Tamanhos das camadas ocultas da MLP (default: 128 64 32).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-3,
        help="Taxa de aprendizado do otimizador Adam.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de épocas de treinamento.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Tamanho do mini-batch durante o treinamento.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporção dos dados destinada ao conjunto de teste.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente de aleatoriedade para reprodutibilidade.",
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=None,
        help="Caminho para salvar as previsões (opcional).",
    )

    args = parser.parse_args(argv)
    metrics = run(
        data_dir=args.data_dir,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.random_state,
        output_predictions=args.output_predictions,
    )

    print("Desempenho do regressor profundo:")
    for name, value in metrics.items():
        if name == "r2":
            print(f"- {name.upper()}: {value:.4f}")
        else:
            print(f"- {name.upper()}: {value:.6f}")


if __name__ == "__main__":
    main()
