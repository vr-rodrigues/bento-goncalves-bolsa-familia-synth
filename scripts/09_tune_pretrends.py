from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

from bg_bolsa.config import load_config, month_sequence
from bg_bolsa.matching import build_features, match_nearest, units_with_complete_panel
from bg_bolsa.synth import run_synth

import importlib.util


def load_report_module():
    path = ROOT / "scripts" / "08_build_economics_letters_report.py"
    spec = importlib.util.spec_from_file_location("economics_letters_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = load_report_module()
OUT = ROOT / "outputs" / "economics_letters"


def load_baseline_covariates() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "municipal_covariates.csv"
    if not path.exists():
        return pd.DataFrame({"id_municipio_6": []})
    cov = pd.read_csv(path, dtype={"id_municipio": str})
    cov["id_municipio_6"] = cov["id_municipio"].astype(str).str[:6]
    for source, target in [
        ("populacao", "log_populacao_2022"),
        ("pib", "log_pib_2021"),
        ("pib_per_capita", "log_pib_per_capita_2021"),
    ]:
        cov[target] = np.log1p(pd.to_numeric(cov[source], errors="coerce"))
    return cov[
        [
            "id_municipio_6",
            "log_populacao_2022",
            "log_pib_2021",
            "log_pib_per_capita_2021",
        ]
    ].drop_duplicates("id_municipio_6")


def previous_month(anomes: int, offset: int) -> int:
    year = anomes // 100
    month = anomes % 100
    month -= offset
    while month <= 0:
        year -= 1
        month += 12
    return year * 100 + month


def lag_months(intervention_month: int, offsets: list[int], pre_months: list[int]) -> list[int]:
    candidates = [previous_month(intervention_month, offset) for offset in offsets]
    return [month for month in candidates if month in set(pre_months)]


def prefit_metrics(ts: pd.DataFrame, pre_months: list[int]) -> dict[str, float]:
    pre = ts.loc[ts["anomes"].isin(pre_months)].copy()
    ma_gap = pre["effect_value"].astype(float).rolling(3, min_periods=1).mean()
    return {
        "pre_rmspe_index": float((pre["effect_index"].astype(float).pow(2).mean()) ** 0.5),
        "pre_rmspe_value": float((pre["effect_value"].astype(float).pow(2).mean()) ** 0.5),
        "pre_rmspe_value_ma3": float((ma_gap.pow(2).mean()) ** 0.5),
        "pre_mean_abs_value_ma3": float(ma_gap.abs().mean()),
        "pre_max_abs_value_ma3": float(ma_gap.abs().max()),
        "pre_last_abs_value_ma3": float(abs(ma_gap.iloc[-1])),
    }


def evaluate_spec(
    outcome_key: str,
    outcome_spec: dict,
    pool_key: str,
    pool_spec: dict,
    lagged_y_months: list[int],
    k: int,
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    treated_id: str,
    panels: dict[str, pd.DataFrame],
    baseline_covariates: pd.DataFrame,
    add_baseline_covariates: bool,
) -> dict:
    panel = panels.setdefault(outcome_spec["panel"], REPORT.load_panel(outcome_spec["panel"]))
    value_col = outcome_spec["value_col"]
    scale_mode = outcome_spec.get("scale_mode", "index")
    require_positive = scale_mode == "index"
    complete_units = units_with_complete_panel(
        panel, all_months, value_col=value_col, require_positive=require_positive
    )
    features = build_features(
        panel[panel["id_municipio_6"].isin(complete_units)],
        pre_months,
        lagged_y_months=lagged_y_months,
        value_col=value_col,
        auxiliary_col="beneficio_medio" if outcome_spec["panel"] == "bolsa" else None,
        require_positive=require_positive,
    )
    labels = panel[["id_municipio_6", "municipio", "sigla_uf"]].drop_duplicates()
    features = features.merge(labels, on=["id_municipio_6", "municipio", "sigla_uf"], how="left")
    if add_baseline_covariates:
        features = features.merge(baseline_covariates, on="id_municipio_6", how="left")
    candidate_ids = REPORT.build_candidate_ids(panel, complete_units, treated_id, pool_spec["ufs"])
    matched = match_nearest(features, treated_id, candidate_ids, k=min(k, len(candidate_ids)))
    donor_ids = matched["id_municipio_6"].tolist()
    result = run_synth(
        panel,
        treated_id,
        donor_ids,
        pre_months,
        all_months,
        baseline_month,
        value_col=value_col,
        value_label=outcome_key,
        scale_mode=scale_mode,
    )
    row = {
        "outcome": outcome_key,
        "pool": pool_key,
        "k": len(donor_ids),
        "lags": ",".join(str(month) for month in lagged_y_months) if lagged_y_months else "none",
        "baseline_covariates": "pop_pib" if add_baseline_covariates else "none",
        "last_effect": result["metrics"]["last_effect_value"],
        "mean_post_effect": result["metrics"]["mean_post_effect_value"],
        "post_pre_ratio": result["metrics"]["post_pre_rmspe_ratio"],
    }
    row.update(prefit_metrics(result["timeseries"], pre_months))
    return row


def main() -> None:
    config = load_config()
    treated_id = config["treatment"]["id_municipio_mds"]
    first_pre_month = int(config["treatment"].get("first_pre_month", 202303))
    intervention_month = int(config["treatment"]["intervention_month"])
    baseline_month = int(config["treatment"]["last_pre_month"])
    post_end = int(config["treatment"]["post_end_month"])
    pre_months = month_sequence(first_pre_month, baseline_month)

    lag_specs = {
        "current_all_pre": pre_months,
        "no_extra_lags": [],
        "t1_t2_t5_t10": lag_months(intervention_month, [1, 2, 5, 10], pre_months),
        "t1_t2_t3_t6": lag_months(intervention_month, [1, 2, 3, 6], pre_months),
        "t1_t3_t6_t12": lag_months(intervention_month, [1, 3, 6, 12], pre_months),
        "t1_t4_t7_t10": lag_months(intervention_month, [1, 4, 7, 10], pre_months),
    }
    rows: list[dict] = []
    panels: dict[str, pd.DataFrame] = {}
    baseline_covariates = load_baseline_covariates()
    for outcome_key in REPORT.MAIN_OUTCOMES:
        outcome_spec = config["outcomes"][outcome_key]
        outcome_post_end = int(outcome_spec.get("post_end_month", post_end))
        all_months = month_sequence(first_pre_month, outcome_post_end)
        for pool_key, pool_spec in REPORT.POOLS.items():
            if pool_key not in {"south", "rs"}:
                continue
            for spec_name, lags in lag_specs.items():
                covariate_modes = [False, True] if not baseline_covariates.empty else [False]
                for add_covariates in covariate_modes:
                    label = f"{spec_name}+pop_pib" if add_covariates else spec_name
                    try:
                        row = evaluate_spec(
                            outcome_key,
                            outcome_spec,
                            pool_key,
                            pool_spec,
                            lags,
                            int(pool_spec["k"]),
                            pre_months,
                            all_months,
                            baseline_month,
                            treated_id,
                            panels,
                            baseline_covariates,
                            add_covariates,
                        )
                        row["spec"] = label
                        rows.append(row)
                    except Exception as exc:
                        rows.append(
                            {
                                "outcome": outcome_key,
                                "pool": pool_key,
                                "spec": label,
                                "lags": ",".join(str(month) for month in lags) if lags else "none",
                                "baseline_covariates": "pop_pib" if add_covariates else "none",
                                "error": str(exc),
                            }
                        )
    out = pd.DataFrame(rows)
    if "error" not in out.columns:
        out["error"] = pd.NA
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "spec_search_pretrends.csv", index=False)

    ranked = (
        out[out["pool"].eq("south") & out["error"].isna()]
        .sort_values(["outcome", "pre_rmspe_value_ma3", "pre_rmspe_index"])
        .groupby("outcome", as_index=False)
        .head(3)
    )
    print(ranked[[
        "outcome",
        "spec",
        "lags",
        "pre_rmspe_index",
        "pre_rmspe_value_ma3",
        "pre_last_abs_value_ma3",
        "last_effect",
        "post_pre_ratio",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
