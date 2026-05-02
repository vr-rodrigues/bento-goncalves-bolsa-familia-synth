from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bg_bolsa.config import load_config, month_sequence
from bg_bolsa.matching import build_features, match_nearest, units_with_complete_panel
from bg_bolsa.reporting import write_report, write_svg_timeseries
from bg_bolsa.synth import placebo_ratios, run_synth


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
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    matched = match_nearest(features, treated_id, candidate_ids, k=k)
    donor_ids = matched["id_municipio_6"].tolist()
    result = run_synth(panel, treated_id, donor_ids, pre_months, all_months, baseline_month)
    result = attach_labels(result, matched)
    placebos = placebo_ratios(panel, donor_ids, pre_months, all_months, baseline_month)
    return result, matched, placebos


def main() -> None:
    config = load_config()
    treated_id = config["treatment"]["id_municipio_mds"]
    first_pre_month = int(config["treatment"].get("first_pre_month", 202303))
    intervention_month = int(config["treatment"]["intervention_month"])
    baseline_month = int(config["treatment"]["last_pre_month"])
    post_end = int(config["treatment"]["post_end_month"])
    pre_months = month_sequence(first_pre_month, baseline_month)
    all_months = month_sequence(first_pre_month, post_end)

    panel_path = ROOT / "data" / "processed" / "bolsa_familia_municipio_panel.csv"
    if not panel_path.exists():
        raise SystemExit("Run scripts/01_collect_mds_bolsa.py first.")
    panel = pd.read_csv(panel_path, dtype={"id_municipio_6": str})
    panel["id_municipio_6"] = panel["id_municipio_6"].str.zfill(6)

    complete_units = units_with_complete_panel(panel, all_months)
    lagged_y_months = (
        pre_months if bool(config["matching"].get("use_all_pre_treatment_y_lags", True)) else []
    )
    features = build_features(
        panel[panel["id_municipio_6"].isin(complete_units)],
        pre_months,
        lagged_y_months=lagged_y_months,
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

    national, national_matched, national_placebos = run_pool(
        panel,
        features,
        treated_id,
        candidates_national,
        int(config["matching"]["national_k"]),
        pre_months,
        all_months,
        baseline_month,
    )
    rs, rs_matched, rs_placebos = run_pool(
        panel,
        features,
        treated_id,
        candidates_rs,
        int(config["matching"]["rs_k"]),
        pre_months,
        all_months,
        baseline_month,
    )

    tables_dir = ROOT / "outputs" / "tables"
    figures_dir = ROOT / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for label, result, matched, placebos in [
        ("national", national, national_matched, national_placebos),
        ("rs", rs, rs_matched, rs_placebos),
    ]:
        result["timeseries"].to_csv(tables_dir / f"bf_synth_{label}_timeseries.csv", index=False)
        result["weights"].merge(result["donor_labels"], on="id_municipio_6", how="left").to_csv(
            tables_dir / f"bf_synth_{label}_weights.csv", index=False
        )
        matched.to_csv(tables_dir / f"bf_match_{label}.csv", index=False)
        placebos.to_csv(tables_dir / f"bf_placebos_{label}.csv", index=False)
        write_svg_timeseries(
            result["timeseries"],
            figures_dir / f"bf_synth_{label}.svg",
            f"Bolsa Familia - {label}",
            intervention_month,
        )

    write_report(ROOT / "reports" / "initial_results.md", national, rs, national_placebos, rs_placebos)
    from bg_bolsa.reporting import write_pdf_report

    write_pdf_report(
        ROOT / "reports" / "initial_results.pdf",
        national,
        rs,
        national_placebos,
        rs_placebos,
    )
    print((ROOT / "reports" / "initial_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
