"""Build and validate the leakage-aware feature dataset.
Run from the repository root with::
    python -m src.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from iapws import IAPWS97

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
GENERATORS = ("gen1", "gen2", "gen3")
SENSOR_COLUMNS = (
    "load_MW", "vent_pressure_bar", "steam_flow_th", "scrubber_temp_C",
    "scrubber_pressure_bar", "lh_inlet_temp_C", "rh_inlet_temp_C",
    "conductivity_uMHO", "chest_pressure_barg", "exhaust_pressure_bara",
    "exhaust_temp_C",
)

def load_clean_unit(generator: str) -> pd.DataFrame:
    """Load one unit, enforce types, ordering, uniqueness, and required fields."""
    path = PROCESSED / f"{generator}_clean.csv"
    frame = pd.read_csv(path)
    missing = {"date", *SENSOR_COLUMNS}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    # Preserve source row order. The workbook contains multiple readings per date
    # and a handful of malformed year values, so sorting would corrupt chronology.
    frame = frame.reset_index(drop=True)
    for column in SENSOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(SENSOR_COLUMNS)].isna().any().any():
        raise ValueError(f"{path.name} contains missing/non-numeric cleaned values")
    return frame


def thermodynamic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate internally consistent saturated-steam proxy properties.

    Sensor pressure/temperature pairs are not forced into IAPWS together because
    the observed pairs are not guaranteed to describe the same equilibrium state.
    The calculation therefore uses measured temperature and saturated states.
    Brine flow assumes separator vapour quality x=0.20; this is a documented proxy,
    not a calibrated double-flash mass balance.
    """
    dead = IAPWS97(T=298.15, P=0.101325)
    rows: list[dict[str, float]] = []
    separator_quality = 0.20
    for row in frame.itertuples(index=False):
        sep_liq = IAPWS97(T=row.scrubber_temp_C + 273.15, x=0)
        inlet = IAPWS97(T=(row.lh_inlet_temp_C + row.rh_inlet_temp_C) / 2 + 273.15, x=1)
        exhaust_liq = IAPWS97(T=row.exhaust_temp_C + 273.15, x=0)
        exhaust_vap = IAPWS97(T=row.exhaust_temp_C + 273.15, x=1)
        quality = np.clip(
            (inlet.s - exhaust_liq.s) / (exhaust_vap.s - exhaust_liq.s), 0.80, 1.0
        )
        h_exhaust = exhaust_liq.h + quality * (exhaust_vap.h - exhaust_liq.h)
        mass_steam = row.steam_flow_th / 3.6  # tonne/hour -> kg/s
        ideal_work = mass_steam * max(inlet.h - h_exhaust, 0.0) / 1000
        inlet_exergy = (inlet.h - dead.h) - 298.15 * (inlet.s - dead.s)
        exergy_in = mass_steam * inlet_exergy / 1000
        mass_brine = mass_steam * (1 - separator_quality) / separator_quality
        brine_exergy = mass_brine * (
            (sep_liq.h - dead.h) - 298.15 * (sep_liq.s - dead.s)
        ) / 1000
        rows.append({
            "h3_kJ_kg": inlet.h,
            "h5_kJ_kg": h_exhaust,
            "h2_brine_kJ_kg": sep_liq.h,
            "x5_exhaust_quality": quality,
            "W_isentropic_MW": ideal_work,
            "turbine_isentropic_eff": np.clip(row.load_MW / ideal_work, 0, 1) if ideal_work else np.nan,
            "exergy_in_MW": exergy_in,
            "exergy_efficiency": row.load_MW / exergy_in if exergy_in else np.nan,
            "brine_exergy_MW": brine_exergy,
        })
    return pd.DataFrame(rows, index=frame.index)


def build_dataset() -> pd.DataFrame:
    units = {name: load_clean_unit(name) for name in GENERATORS}
    dates = units["gen1"]["date"]
    for name, unit in units.items():
        if not dates.equals(unit["date"]):
            raise ValueError(f"{name} dates do not align exactly with gen1")

    output = pd.DataFrame({"observation_id": np.arange(1, len(dates) + 1), "date": dates})
    for name, unit in units.items():
        for column in SENSOR_COLUMNS:
            output[f"{name}_{column}"] = unit[column]
        output[f"{name}_inlet_temp_avg"] = (
            unit["lh_inlet_temp_C"] + unit["rh_inlet_temp_C"]
        ) / 2
        output[f"{name}_inlet_temp_asymmetry"] = (
            unit["lh_inlet_temp_C"] - unit["rh_inlet_temp_C"]
        ).abs()
        # Chest pressure is gauge; convert to absolute before subtracting exhaust bara.
        output[f"{name}_pressure_drop_bar"] = (
            unit["chest_pressure_barg"] + 1.01325 - unit["exhaust_pressure_bara"]
        )
        output[f"{name}_steam_utilization"] = unit["load_MW"] / unit["steam_flow_th"]

        # All rolling windows are shifted first: current/future target values never leak in.
        output[f"{name}_load_lag1"] = unit["load_MW"].shift(1)
        output[f"{name}_steam_lag1"] = unit["steam_flow_th"].shift(1)
        output[f"{name}_conductivity_lag1"] = unit["conductivity_uMHO"].shift(1)
        output[f"{name}_load_roll3"] = unit["load_MW"].shift(1).rolling(3).mean()
        output[f"{name}_load_roll3_std"] = unit["load_MW"].shift(1).rolling(3).std()
        output[f"{name}_steam_roll3"] = unit["steam_flow_th"].shift(1).rolling(3).mean()
        output[f"{name}_conductivity_roll3"] = unit["conductivity_uMHO"].shift(1).rolling(3).mean()

        thermo = thermodynamic_features(unit)
        for column in thermo:
            output[f"{name}_{column}"] = thermo[column]

    output["post_regime_shift"] = (output["date"] >= "2019-01-01").astype(int)
    output["month_sin"] = np.sin(2 * np.pi * output["date"].dt.month / 12)
    output["month_cos"] = np.cos(2 * np.pi * output["date"].dt.month / 12)
    output["year_offset"] = output["date"].dt.year - output["date"].dt.year.min()
    return output


def validate_dataset(frame: pd.DataFrame) -> dict[str, object]:
    pressure_columns = [c for c in frame if c.endswith("pressure_drop_bar")]
    quality_columns = [c for c in frame if c.endswith("x5_exhaust_quality")]
    eff_columns = [c for c in frame if c.endswith("exergy_efficiency")]
    report = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "date_min": frame["date"].min().isoformat(),
        "date_max": frame["date"].max().isoformat(),
        "repeated_date_observations": int(frame["date"].duplicated().sum()),
        "non_monotonic_date_transitions": int((frame["date"].diff().dt.days < 0).sum()),
        "suspect_year_rows_after_2019": int((frame["date"].dt.year > 2019).sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "expected_history_missing_cells": 45,
        "nonpositive_pressure_drops": int((frame[pressure_columns] <= 0).sum().sum()),
        "quality_outside_0_1": int(((frame[quality_columns] < 0) | (frame[quality_columns] > 1)).sum().sum()),
        "exergy_efficiency_outside_0_1": int(((frame[eff_columns] < 0) | (frame[eff_columns] > 1)).sum().sum()),
        "notes": [
            "The first three observations contain expected lag/rolling NaNs.",
            "Repeated dates are retained because the source has multiple operating readings per day.",
            "Source row order is authoritative because workbook date locale/year defects prevent safe sorting.",
            "Exergy targets are physics-derived proxies and are not independent plant measurements.",
            "Brine exergy assumes separator vapour quality of 0.20 and requires later calibration.",
        ],
    }
    return report


def main() -> None:
    frame = build_dataset()
    output = PROCESSED / "olkaria_features_engineered.csv"
    frame.to_csv(output, index=False)
    report = validate_dataset(frame)
    report_path = PROCESSED / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {frame.shape} to {output}")

if __name__ == "__main__":
    main()
