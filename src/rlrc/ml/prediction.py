"""Optional 3-second risk forecast; split by whole scenario, never neighboring rows."""

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from rlrc.ml.evaluation import save_json
from rlrc.ml.features import FEATURE_VERSION, FeatureBuilder, apply_action
from rlrc.ml.simulator import SimConfig, Simulator


def dataset(seeds: range, config: SimConfig, horizon: int = 3) -> tuple[Any, Any]:
    xs, ys = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed + 900000)
        sim = Simulator(config, seed)
        builder = FeatureBuilder(config.max_rate, config.target, config.inflight_limit)
        observations, unhealthy = [], []
        # Persistent random actions produce varied operating points, not labels from AIMD.
        action = 3
        for step in range(config.duration):
            if step % 5 == 0:
                action = int(rng.integers(0, 6))
            rate = apply_action(sim.rate, action, config.min_rate, config.max_rate)
            snapshot, result = sim.advance(rate)
            observations.append(builder.build(snapshot))
            unhealthy.append(
                bool(
                    result["backend_errors"] + result["timeouts"]
                    > 0.02 * max(1, result["admitted"])
                    or result["late"] > 0.1 * max(1, result["success"])
                )
            )
        for at in range(len(observations) - horizon):
            xs.append(observations[at])
            ys.append(any(unhealthy[at + 1 : at + 1 + horizon]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


def train_predictor(output: Path, episodes: int = 100) -> Path:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    begin = time.perf_counter()
    config = SimConfig()
    train_x, train_y = dataset(range(1000, 1000 + episodes), config)
    valid_x, valid_y = dataset(range(30000, 30020), config)
    test_x, test_y = dataset(range(50000, 50020), config)
    classifier = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=15, l2_regularization=1, random_state=17
    )
    classifier.fit(train_x, train_y)
    # Calibration data are entirely separate scenarios from estimator fitting.
    from sklearn.frozen import FrozenEstimator

    calibrated = CalibratedClassifierCV(FrozenEstimator(classifier), method="sigmoid")
    calibrated.fit(valid_x, valid_y)
    pred = calibrated.predict_proba(test_x)[:, 1]
    artifact = output / "predictor.joblib"
    joblib.dump(calibrated, artifact)
    save_json(
        output / "metadata.json",
        {
            "target": (
                "any next-3s interval: late-success fraction > .1 OR backend-failure/admitted > .02"
            ),
            "horizon": 3,
            "train_seeds": [1000, 1000 + episodes - 1],
            "calibration_seeds": [30000, 30019],
            "test_seeds": [50000, 50019],
            "train_rows": len(train_y),
            "features": FeatureBuilder(config.max_rate, config.target, config.inflight_limit).names,
            "simulator": config.as_dict(),
            "feature_version": FEATURE_VERSION,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "test_rows": len(test_y),
            "prevalence": float(test_y.mean()),
            "roc_auc": float(roc_auc_score(test_y, pred)),
            "average_precision": float(average_precision_score(test_y, pred)),
            "brier": float(brier_score_loss(test_y, pred)),
            "constant_brier": float(brier_score_loss(test_y, np.full(len(test_y), train_y.mean()))),
            "persistence_brier": float(
                brier_score_loss(
                    test_y, ((test_x[:, 6] > 0.25) | (test_x[:, 7] > 0.02)).astype(float)
                )
            ),
            "log_loss": float(log_loss(test_y, pred)),
            "seconds": time.perf_counter() - begin,
            "limitation": (
                "Forecast under the sampled behavior policy, not a causal "
                "counterfactual for a candidate rate; abrupt exogenous failures "
                "may be unpredictable."
            ),
        },
    )
    np.savez_compressed(output / "heldout-predictions.npz", y=test_y, probability=pred)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    print(train_predictor(args.output, args.episodes))
