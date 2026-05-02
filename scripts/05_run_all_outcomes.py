from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bg_bolsa.config import load_config, month_sequence
from bg_bolsa.matching import build_features, match_nearest, units_with_complete_panel
from bg_bolsa.reporting import (
    synth_p_value,
    write_multi_outcome_markdown,
    write_multi_outcome_pdf,
    write_svg_timeseries,
)
from bg_bolsa.synth import in_time_placebo, leave_one_out, placebo_ratios, run_synth


def attach_labels(result: dict, labels: pd.DataFrame) -> dict:
    result["donor_labels"] = labels[["id_municipio_6", "municipio", "sigla_uf"]].drop_duplicates()
    return result


def run_pool(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    treated_id: str,
    candidate_ids: list[str],
    k: int,
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    value_col: str,
    value_label: str,
    fake_pre_months: list[int],
    fake_post_months: list[int],
    fake_baseline_month: int,
    scale_mode: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    matched = match_nearest(features, treated_id, candidate_ids, k=k)
    donor_ids = matched["id_municipio_6"].tolist()
    result = run_synth(
        panel,
        treated_id,
        donor_ids,
        pre_months,
        all_months,
        baseline_month,
        value_col=value_col,
        value_label=value_label,
        scale_mode=scale_mode,
    )
    result = attach_labels(result, matched)
    placebos = placebo_ratios(
        panel,
        donor_ids,
        pre_months,
        all_months,
        baseline_month,
        value_col=value_col,
        scale_mode=scale_mode,
    )
    loo = leave_one_out(
        panel,
        treated_id,
        donor_ids,
        result["weights"],
        pre_months,
        all_months,
        baseline_month,
        value_col=value_col,
        scale_mode=scale_mode,
    )
    fake = in_time_placebo(
        panel,
        treated_id,
        donor_ids,
        fake_pre_months,
        fake_post_months,
        fake_baseline_month,
        value_col=value_col,
        scale_mode=scale_mode,
    )
    return result, matched, placebos, loo, fake


def load_panel(kind: str) -> pd.DataFrame:
    if kind == "bolsa":
        path = ROOT / "data" / "processed" / "bolsa_familia_municipio_panel.csv"
    elif kind == "caged":
        path = ROOT / "data" / "processed" / "caged_municipio_panel.csv"
    else:
        raise ValueError(f"Unknown panel kind: {kind}")
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the collection scripts first.")
    panel = pd.read_csv(path, dtype={"id_municipio_6": str})
    panel["id_municipio_6"] = panel["id_municipio_6"].astype(str).str.zfill(6)
    if kind == "caged":
        panel = panel.sort_values(["id_municipio_6", "anomes"]).copy()
        panel["estoque_lag"] = panel.groupby("id_municipio_6")["estoque"].shift(1)
        denom = panel["estoque_lag"].where(panel["estoque_lag"] > 0)
        panel["taxa_admissao"] = 100.0 * panel["admissoes"] / denom
        panel["taxa_desligamento"] = 100.0 * panel["desligamentos"] / denom
        panel["taxa_rotatividade"] = (
            100.0 * (panel["admissoes"] + panel["desligamentos"]) / denom
        )
    return panel


def main() -> None:
    config = load_config()
    treated_id = config["treatment"]["id_municipio_mds"]
    first_pre_month = int(config["treatment"].get("first_pre_month", 202303))
    intervention_month = int(config["treatment"]["intervention_month"])
    baseline_month = int(config["treatment"]["last_pre_month"])
    post_end = int(config["treatment"]["post_end_month"])
    pre_months = month_sequence(first_pre_month, baseline_month)
    all_months = month_sequence(first_pre_month, post_end)
    fake_pre_months = month_sequence(first_pre_month, 202404)
    fake_post_months = month_sequence(202405, baseline_month)
    fake_baseline_month = 202404

    tables_dir = ROOT / "outputs" / "tables"
    figures_dir = ROOT / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    analyses: list[dict] = []
    summary_rows: list[dict] = []

    for outcome_key, spec in config["outcomes"].items():
        panel = load_panel(spec["panel"])
        value_col = spec["value_col"]
        label = spec["label"]
        scale_mode = spec.get("scale_mode", "index")
        require_positive = scale_mode == "index"
        complete_units = units_with_complete_panel(
            panel,
            all_months,
            value_col=value_col,
            require_positive=require_positive,
        )
        lagged_y_months = (
            pre_months if bool(config["matching"].get("use_all_pre_treatment_y_lags", True)) else []
        )
        aux_col = "beneficio_medio" if spec["panel"] == "bolsa" else None
        features = build_features(
            panel[panel["id_municipio_6"].isin(complete_units)],
            pre_months,
            lagged_y_months=lagged_y_months,
            value_col=value_col,
            auxiliary_col=aux_col,
            require_positive=require_positive,
        )
        labels = panel[["id_municipio_6", "municipio", "sigla_uf"]].drop_duplicates()
        features = features.merge(labels, on=["id_municipio_6", "municipio", "sigla_uf"], how="left")

        candidates_national = [u for u in complete_units if u != treated_id]
        candidates_rs = [
            u
            for u in complete_units
            if u != treated_id
            and not panel.loc[panel["id_municipio_6"] == u, "sigla_uf"].dropna().empty
            and panel.loc[panel["id_municipio_6"] == u, "sigla_uf"].dropna().iloc[0] == "RS"
        ]

        pools = {}
        for pool_key, candidate_ids, k in [
            ("national", candidates_national, int(config["matching"]["national_k"])),
            ("rs", candidates_rs, int(config["matching"]["rs_k"])),
        ]:
            result, matched, placebos, loo, fake = run_pool(
                panel,
                features,
                treated_id,
                candidate_ids,
                k,
                pre_months,
                all_months,
                baseline_month,
                value_col,
                outcome_key,
                fake_pre_months,
                fake_post_months,
                fake_baseline_month,
                scale_mode,
            )
            pools[pool_key] = {
                "result": result,
                "matched": matched,
                "placebos": placebos,
                "leave_one_out": loo,
                "in_time": fake,
            }

            prefix = f"{outcome_key}_{pool_key}"
            result["timeseries"].to_csv(tables_dir / f"synth_{prefix}_timeseries.csv", index=False)
            result["weights"].merge(result["donor_labels"], on="id_municipio_6", how="left").to_csv(
                tables_dir / f"synth_{prefix}_weights.csv", index=False
            )
            matched.to_csv(tables_dir / f"match_{prefix}.csv", index=False)
            placebos.to_csv(tables_dir / f"placebos_{prefix}.csv", index=False)
            loo.to_csv(tables_dir / f"leave_one_out_{prefix}.csv", index=False)
            fake["timeseries"].to_csv(tables_dir / f"in_time_placebo_{prefix}.csv", index=False)
            write_svg_timeseries(
                result["timeseries"],
                figures_dir / f"synth_{prefix}.svg",
                f"{label} - {pool_key}",
                intervention_month,
                "Indice: outubro de 2024 = 100"
                if scale_mode == "index"
                else "Nivel mensal em unidades originais",
            )

            metrics = result["metrics"]
            fake_metrics = fake["metrics"]
            summary_rows.append(
                {
                    "outcome": outcome_key,
                    "pool": pool_key,
                    "scale_mode": scale_mode,
                    "pre_rmspe_index": metrics["pre_rmspe_index"],
                    "post_pre_rmspe_ratio": metrics["post_pre_rmspe_ratio"],
                    "p_value_space_placebo": synth_p_value(metrics, placebos),
                    "mean_post_effect_index": metrics["mean_post_effect_index"],
                    "mean_post_effect_value": metrics["mean_post_effect_value"],
                    "last_actual_value": metrics["last_actual_value"],
                    "last_synth_value": metrics["last_synth_value"],
                    "last_effect_value": metrics["last_effect_value"],
                    "fake_time_placebo_mean_effect_index": fake_metrics[
                        "mean_post_effect_index"
                    ],
                    "fake_time_placebo_last_effect_index": fake_metrics["last_effect_index"],
                    "leave_one_out_last_effect_min": None
                    if loo.empty
                    else loo["last_effect_value"].min(),
                    "leave_one_out_last_effect_max": None
                    if loo.empty
                    else loo["last_effect_value"].max(),
                }
            )

        analyses.append({"key": outcome_key, "label": label, "pools": pools})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "synth_tests_summary.csv", index=False)
    write_multi_outcome_markdown(ROOT / "reports" / "all_outcomes_results.md", analyses)
    write_multi_outcome_pdf(ROOT / "reports" / "all_outcomes_results.pdf", analyses)
    print((ROOT / "reports" / "all_outcomes_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
