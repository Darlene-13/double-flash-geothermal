"""Train chronological baselines and a compact multi-output ANN surrogate."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "olkaria_features_engineered.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
GENERATORS = ("gen1", "gen2", "gen3")


def safe_features(frame: pd.DataFrame, generator: str) -> list[str]:
    """Return only inputs available without knowing either same-row target."""
    allowed_suffixes = (
        "vent_pressure_bar", "steam_flow_th", "scrubber_temp_C",
        "scrubber_pressure_bar", "conductivity_uMHO", "chest_pressure_barg",
        "exhaust_pressure_bara", "exhaust_temp_C", "inlet_temp_avg",
        "inlet_temp_asymmetry", "pressure_drop_bar", "load_lag1",
        "steam_lag1", "conductivity_lag1", "load_roll3", "load_roll3_std",
        "steam_roll3", "conductivity_roll3", "W_isentropic_MW",
        "exergy_in_MW", "h2_brine_kJ_kg", "h3_kJ_kg", "h5_kJ_kg",
        "x5_exhaust_quality", "brine_exergy_MW",
    )
    features = [f"{generator}_{suffix}" for suffix in allowed_suffixes]
    features += ["post_regime_shift", "month_sin", "month_cos", "year_offset"]
    missing = set(features).difference(frame.columns)
    if missing:
        raise ValueError(f"Engineered dataset lacks features: {sorted(missing)}")
    forbidden = ("steam_utilization", "turbine_isentropic_eff", "exergy_efficiency")
    if any(any(token in feature for token in forbidden) for feature in features):
        raise AssertionError("Target-derived leakage feature selected")
    return features


def chronological_slices(size: int) -> tuple[slice, slice, slice]:
    train_end = int(size * 0.70)
    validation_end = int(size * 0.85)
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, size)


def metric_rows(generator: str, model: str, split: str, y: np.ndarray, pred: np.ndarray) -> list[dict]:
    rows = []
    for index, target in enumerate(("load_MW", "exergy_efficiency")):
        rows.append({
            "generator": generator,
            "model": model,
            "split": split,
            "target": target,
            "MAE": mean_absolute_error(y[:, index], pred[:, index]),
            "RMSE": mean_squared_error(y[:, index], pred[:, index]) ** 0.5,
            "R2": r2_score(y[:, index], pred[:, index]),
        })
    return rows


def fit_ann_with_temporal_early_stopping(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
) -> tuple[MLPRegressor, dict[str, list[float]]]:
    """Warm-start one epoch at a time and stop using the later validation block."""
    model = MLPRegressor(
        hidden_layer_sizes=(24, 12), activation="relu", solver="adam",
        alpha=0.01, batch_size=min(32, len(x_train)), learning_rate_init=0.001,
        max_iter=1, warm_start=True, random_state=42,
    )
    history = {"train_mse": [], "validation_mse": []}
    best_model = None
    best_loss = np.inf
    stale_epochs = 0
    for _ in range(500):
        # max_iter=1 is deliberate for external temporal early stopping.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train)
        train_loss = mean_squared_error(y_train, model.predict(x_train))
        val_loss = mean_squared_error(y_val, model.predict(x_val))
        history["train_mse"].append(float(train_loss))
        history["validation_mse"].append(float(val_loss))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 40:
            break
    assert best_model is not None
    return best_model, history


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    # observation_id preserves workbook chronology despite known malformed dates.
    frame = pd.read_csv(DATA, parse_dates=["date"]).sort_values("observation_id").reset_index(drop=True)
    all_metrics: list[dict] = []
    split_summary: dict[str, dict] = {}

    for generator in GENERATORS:
        features = safe_features(frame, generator)
        targets = [f"{generator}_load_MW", f"{generator}_exergy_efficiency"]
        modelling = frame[["observation_id", "date", *features, *targets]].dropna().reset_index(drop=True)
        # Remove physically invalid derived targets rather than teaching the model
        # an exergy efficiency above unity (one such row exists in the source).
        modelling = modelling[
            (modelling[targets[0]] > 0)
            & modelling[targets[1]].between(0, 1, inclusive="both")
        ].reset_index(drop=True)
        train, validation, test = chronological_slices(len(modelling))
        x = modelling[features]
        y = modelling[targets].to_numpy()
        split_summary[generator] = {
            name: {
                "rows": len(modelling.iloc[part]),
                "observation_id_start": int(modelling.iloc[part]["observation_id"].min()),
                "observation_id_end": int(modelling.iloc[part]["observation_id"].max()),
                "start": modelling.iloc[part]["date"].min().date().isoformat(),
                "end": modelling.iloc[part]["date"].max().date().isoformat(),
            }
            for name, part in (("train", train), ("validation", validation), ("test", test))
        }

        baseline_models = {
            "mean": DummyRegressor(strategy="mean"),
            "ridge": Ridge(alpha=10.0),
            "random_forest": RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
        }
        for name, estimator in baseline_models.items():
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler() if name == "ridge" else "passthrough"),
                ("model", estimator),
            ])
            pipeline.fit(x.iloc[train], y[train])
            for split_name, part in (("validation", validation), ("test", test)):
                all_metrics.extend(metric_rows(
                    generator, name, split_name, y[part], pipeline.predict(x.iloc[part])
                ))

        x_imputer = SimpleImputer(strategy="median").fit(x.iloc[train])
        x_scaler = StandardScaler().fit(x_imputer.transform(x.iloc[train]))
        y_scaler = StandardScaler().fit(y[train])
        x_scaled = x_scaler.transform(x_imputer.transform(x))
        y_scaled = y_scaler.transform(y)
        ann, history = fit_ann_with_temporal_early_stopping(
            x_scaled[train], y_scaled[train], x_scaled[validation], y_scaled[validation]
        )
        for split_name, part in (("validation", validation), ("test", test)):
            prediction = y_scaler.inverse_transform(ann.predict(x_scaled[part]))
            all_metrics.extend(metric_rows(generator, "ann", split_name, y[part], prediction))
        joblib.dump({
            "generator": generator, "features": features, "targets": targets,
            "x_imputer": x_imputer, "x_scaler": x_scaler,
            "y_scaler": y_scaler, "model": ann,
        }, MODEL_DIR / f"{generator}_ann.joblib")
        (REPORT_DIR / f"{generator}_ann_history.json").write_text(json.dumps(history, indent=2) + "\n")

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    (REPORT_DIR / "chronological_splits.json").write_text(json.dumps(split_summary, indent=2) + "\n")
    print(metrics[metrics["split"] == "test"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
