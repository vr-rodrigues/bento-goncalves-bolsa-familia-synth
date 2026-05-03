from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bg_bolsa.config import load_config, month_sequence
from bg_bolsa.firpo_possebom import confidence_set_for_panel
from bg_bolsa.matching import build_features, match_nearest, units_with_complete_panel
from bg_bolsa.synth import in_time_placebo, leave_one_out, run_synth

PAPER = ROOT / "paper" / "economics_letters_project"
OUT = ROOT / "outputs" / "economics_letters"

SOUTH_UFS = {"RS", "SC", "PR"}
POOLS = {
    "rs": {"label": "RS", "k": 30, "ufs": {"RS"}},
    "south": {"label": "South", "k": 40, "ufs": SOUTH_UFS},
}
DONOR_PREFILTER_K = 80
DONOR_SELECTION_K_GRID = [12, 15, 20, 25]
SPARSE_LAG_OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18]

DEFAULT_MATCHING_SPEC = {"name": "sparse_pre_lags", "lag_offsets": SPARSE_LAG_OFFSETS, "baseline_covariates": False}
OUTCOME_MATCHING_SPECS = {
    "bolsa_familia": {
        "name": "sparse_pre_lags_pop_pib",
        "lag_offsets": SPARSE_LAG_OFFSETS,
        "baseline_covariates": True,
    },
    "emprego_estoque": {
        "name": "sparse_pre_lags_pop_pib",
        "lag_offsets": SPARSE_LAG_OFFSETS,
        "baseline_covariates": True,
    },
    "taxa_admissao": {
        "name": "lags_t1_t2_t5_t10",
        "lag_offsets": [1, 2, 5, 10],
        "baseline_covariates": False,
    },
    "empresas_abertas": {
        "name": "no_extra_lags",
        "lag_offsets": [],
        "baseline_covariates": False,
    },
}

MAIN_OUTCOMES = [
    "bolsa_familia",
    "emprego_estoque",
]

APPENDIX_EXTRA_OUTCOMES: list[str] = []

OUTCOME_LABELS = {
    "bolsa_familia": "Famílias beneficiárias do Bolsa Família",
    "emprego_estoque": "Estoque de vínculos formais",
    "emprego_admissoes": "Admissões formais",
    "emprego_desligamentos": "Desligamentos formais",
    "emprego_saldo": "Saldo líquido de empregos formais",
    "taxa_admissao": "Taxa de admissão formal",
    "taxa_desligamento": "Taxa de desligamento formal",
    "taxa_rotatividade": "Taxa de rotatividade formal",
    "empresas_abertas": "Nascimento de empresas",
}

OUTCOME_SHORT_LABELS = {
    "bolsa_familia": "Bolsa Família",
    "emprego_estoque": "Estoque formal",
    "emprego_admissoes": "Admissões",
    "emprego_desligamentos": "Desligamentos",
    "emprego_saldo": "Saldo formal",
    "taxa_admissao": "Taxa admissão",
    "taxa_desligamento": "Taxa desligamento",
    "taxa_rotatividade": "Taxa rotatividade",
    "empresas_abertas": "Empresas abertas",
}

Y_AXIS_LABELS = {
    "bolsa_familia": "Famílias (n)",
    "emprego_estoque": "Vínculos (n)",
    "emprego_admissoes": "Admissões (n/mês)",
    "emprego_desligamentos": "Desligamentos (n/mês)",
    "emprego_saldo": "Saldo (n/mês)",
    "taxa_admissao": "Taxa (%)",
    "taxa_desligamento": "Taxa (%)",
    "taxa_rotatividade": "Taxa (%)",
    "empresas_abertas": "Empresas (n/mês)",
}

EFFECT_AXIS_LABELS = {
    "bolsa_familia": "Gap",
    "emprego_estoque": "Gap",
    "emprego_admissoes": "Gap",
    "emprego_desligamentos": "Gap",
    "emprego_saldo": "Gap",
    "taxa_admissao": "Gap",
    "taxa_desligamento": "Gap",
    "taxa_rotatividade": "Gap",
    "empresas_abertas": "Gap",
}

OUTCOME_TABLE_LABELS = {
    "bolsa_familia": "Bolsa Família (famílias)",
    "emprego_estoque": "Estoque formal (vínculos)",
    "emprego_admissoes": "Admissões formais (Quantidade/mês)",
    "taxa_admissao": "Taxa de admissão formal (p.p.)",
    "empresas_abertas": "Nascimento de empresas (Quantidade/mês)",
}

OUTCOME_FIGURE_TITLES = {
    "bolsa_familia": "Famílias beneficiárias do Bolsa Família (Quantidade)",
    "emprego_estoque": "Estoque de vínculos formais (Quantidade)",
    "emprego_admissoes": "Admissões formais (Quantidade/mês)",
    "taxa_admissao": "Taxa de admissão formal (p.p.)",
    "empresas_abertas": "Nascimento de empresas (Quantidade/mês)",
}


def tex_escape(value: object) -> str:
    text = str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        text = text.replace(old, new)
    return text


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def fmt0(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):,.0f}".replace(",", ".")


def fmt_effect(value: float | int | None, outcome_key: str) -> str:
    if value is None or pd.isna(value):
        return "--"
    value = float(value)
    if outcome_key.startswith("taxa_"):
        return f"{value:.2f}"
    if abs(value) < 10:
        return f"{value:.1f}"
    return fmt0(value)


def month_label(anomes: int) -> str:
    year = int(anomes) // 100
    month = int(anomes) % 100
    return f"{year}m{month}"


def previous_month(anomes: int, offset: int) -> int:
    year = int(anomes) // 100
    month = int(anomes) % 100
    month -= int(offset)
    while month <= 0:
        year -= 1
        month += 12
    return year * 100 + month


def matching_lag_months(outcome_key: str, intervention_month: int, pre_months: list[int]) -> list[int]:
    spec = OUTCOME_MATCHING_SPECS.get(outcome_key, DEFAULT_MATCHING_SPEC)
    offsets = spec.get("lag_offsets", "all")
    if offsets == "all":
        return pre_months
    if not offsets:
        return []
    pre_set = set(pre_months)
    return [previous_month(intervention_month, offset) for offset in offsets if previous_month(intervention_month, offset) in pre_set]


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


def axis_tick_label(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    abs_value = abs(value)
    if abs_value < 1e-8:
        return "0"
    if abs_value >= 100:
        return f"{value:,.0f}".replace(",", ".")
    if abs_value >= 10:
        return f"{value:.0f}"
    if abs_value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def nice_tick_step(span: float, target_ticks: int = 4, minimum: float = 100.0) -> float:
    if span <= 0 or pd.isna(span):
        return minimum
    raw = max(float(span) / max(target_ticks, 1), minimum)
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiplier in [1, 2, 5, 10]:
        step = multiplier * magnitude
        if step >= raw:
            return step
    return 10 * magnitude


def rounded_axis_ticks(
    y_low: float,
    y_high: float,
    *,
    include_zero: bool = False,
    minimum_step: float = 100.0,
    target_ticks: int = 4,
) -> list[float]:
    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_high <= y_low:
        return []
    step = nice_tick_step(y_high - y_low, target_ticks=target_ticks, minimum=minimum_step)
    first = math.ceil(y_low / step) * step
    last = math.floor(y_high / step) * step
    ticks = list(np.arange(first, last + step * 0.5, step))
    if include_zero and y_low < 0 < y_high and not any(abs(t) < 1e-8 for t in ticks):
        ticks.append(0.0)
        ticks = sorted(ticks)
    return [float(t) for t in ticks]


def moving_average(values: pd.Series, window: int) -> pd.Series:
    return values.astype(float).rolling(window, min_periods=1).mean()


def t_stat(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(x) <= 1:
        return float("nan")
    sd = float(x.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(x.mean() / (sd / math.sqrt(len(x))))


def add_caged_rates(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["id_municipio_6", "anomes"]).copy()
    panel["estoque_lag"] = panel.groupby("id_municipio_6")["estoque"].shift(1)
    denom = panel["estoque_lag"].where(panel["estoque_lag"] > 0)
    panel["taxa_admissao"] = 100.0 * panel["admissoes"] / denom
    panel["taxa_desligamento"] = 100.0 * panel["desligamentos"] / denom
    panel["taxa_rotatividade"] = 100.0 * (panel["admissoes"] + panel["desligamentos"]) / denom
    return panel


def load_panel(kind: str) -> pd.DataFrame:
    if kind == "bolsa":
        path = ROOT / "data" / "processed" / "bolsa_familia_municipio_panel.csv"
    elif kind == "caged":
        path = ROOT / "data" / "processed" / "caged_municipio_panel.csv"
    elif kind == "cnpj":
        cnpj_path = ROOT / "data" / "processed" / "cnpj_births_municipio_monthly.csv"
        labels_path = ROOT / "data" / "processed" / "caged_municipio_panel.csv"
        panel = pd.read_csv(cnpj_path, dtype={"id_municipio": str})
        panel["id_municipio_6"] = panel["id_municipio"].astype(str).str[:6]
        panel["anomes"] = panel["anomes"].astype(int)
        labels = (
            pd.read_csv(labels_path, dtype={"id_municipio_6": str})[
                ["id_municipio_6", "municipio", "sigla_uf", "uf", "regiao"]
            ]
            .drop_duplicates("id_municipio_6")
        )
        labels = labels[labels["sigla_uf"].isin(SOUTH_UFS)].copy()
        months = pd.DataFrame({"anomes": month_sequence(int(panel["anomes"].min()), int(panel["anomes"].max()))})
        labels = labels.assign(_key=1)
        months = months.assign(_key=1)
        grid = labels.merge(months, on="_key").drop(columns="_key")
        value_cols = ["empresas_abertas", "empresas_abertas_ativas_no_snapshot", "matrizes_abertas"]
        panel = grid.merge(
            panel[["id_municipio_6", "anomes", *value_cols]],
            on=["id_municipio_6", "anomes"],
            how="left",
        )
        for col in value_cols:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)
        return panel
    else:
        raise ValueError(kind)
    panel = pd.read_csv(path, dtype={"id_municipio_6": str})
    panel["id_municipio_6"] = panel["id_municipio_6"].astype(str).str.zfill(6)
    if kind == "caged":
        panel = add_caged_rates(panel)
    return panel


def build_candidate_ids(panel: pd.DataFrame, complete_units: list[str], treated_id: str, ufs: set[str]) -> list[str]:
    labels = panel[["id_municipio_6", "sigla_uf"]].dropna().drop_duplicates()
    eligible = set(labels.loc[labels["sigla_uf"].isin(ufs), "id_municipio_6"])
    return [u for u in complete_units if u != treated_id and u in eligible]


def direct_pre_rmspe_ranking(
    panel: pd.DataFrame,
    treated_id: str,
    donor_ids: list[str],
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    value_col: str,
    scale_mode: str,
) -> pd.DataFrame:
    pivot = panel.pivot_table(
        index="anomes", columns="id_municipio_6", values=value_col, aggfunc="first"
    ).sort_index()
    required_units = [treated_id] + donor_ids
    counts = pivot.loc[all_months, required_units].copy()
    if scale_mode == "index":
        baseline = counts.loc[baseline_month]
        scaled = counts.divide(baseline, axis=1) * 100.0
    elif scale_mode == "level":
        scaled = counts.copy()
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    treated_pre = scaled.loc[pre_months, treated_id].astype(float)
    rows = []
    for donor_id in donor_ids:
        donor_pre = scaled.loc[pre_months, donor_id].astype(float)
        gap = treated_pre - donor_pre
        rows.append(
            {
                "id_municipio_6": donor_id,
                "direct_pre_rmspe": float(np.sqrt(np.mean(np.square(gap)))),
            }
        )
    return pd.DataFrame(rows).sort_values("direct_pre_rmspe").reset_index(drop=True)


def select_donor_pool(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    treated_id: str,
    candidate_ids: list[str],
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    value_col: str,
    scale_mode: str,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    prefilter_k = min(DONOR_PREFILTER_K, len(candidate_ids))
    if prefilter_k < min(DONOR_SELECTION_K_GRID):
        matched = match_nearest(features, treated_id, candidate_ids, k=prefilter_k)
        return matched, {
            "donor_selection_order": "matching_distance",
            "donor_selection_k": len(matched),
            "donor_selection_pre_rmspe": float("nan"),
        }

    matched_prefilter = match_nearest(features, treated_id, candidate_ids, k=prefilter_k)
    matched_ids = matched_prefilter["id_municipio_6"].tolist()
    direct_ranking = direct_pre_rmspe_ranking(
        panel,
        treated_id,
        matched_ids,
        pre_months,
        all_months,
        baseline_month,
        value_col,
        scale_mode,
    )
    orderings = {
        "matching_distance": matched_ids,
        "direct_pre_rmspe": direct_ranking["id_municipio_6"].tolist(),
    }
    candidates = []
    for order_name, ordered_ids in orderings.items():
        for k in DONOR_SELECTION_K_GRID:
            if len(ordered_ids) < k:
                continue
            donor_ids = ordered_ids[:k]
            try:
                result = run_synth(
                    panel,
                    treated_id,
                    donor_ids,
                    pre_months,
                    all_months,
                    baseline_month,
                    value_col=value_col,
                    scale_mode=scale_mode,
                )
            except (ValueError, np.linalg.LinAlgError, KeyError):
                continue
            candidates.append(
                {
                    "donor_selection_order": order_name,
                    "donor_selection_k": k,
                    "donor_selection_pre_rmspe": result["metrics"]["pre_rmspe_index"],
                    "donor_ids": donor_ids,
                }
            )
    if not candidates:
        matched = matched_prefilter.head(min(max(DONOR_SELECTION_K_GRID), len(matched_prefilter))).copy()
        return matched, {
            "donor_selection_order": "matching_distance",
            "donor_selection_k": len(matched),
            "donor_selection_pre_rmspe": float("nan"),
        }

    selected = sorted(
        candidates,
        key=lambda row: (row["donor_selection_pre_rmspe"], row["donor_selection_k"]),
    )[0]
    diagnostics = {k: v for k, v in selected.items() if k != "donor_ids"}
    selected_order = pd.DataFrame(
        {
            "id_municipio_6": selected["donor_ids"],
            "donor_selection_rank": range(1, len(selected["donor_ids"]) + 1),
        }
    )
    matched = (
        selected_order.merge(
            matched_prefilter.drop(columns=["donor_selection_rank"], errors="ignore"),
            on="id_municipio_6",
            how="left",
        )
        .merge(direct_ranking, on="id_municipio_6", how="left")
        .merge(labels, on=["id_municipio_6", "municipio", "sigla_uf"], how="left")
    )
    return matched, diagnostics


def placebo_distribution(
    panel: pd.DataFrame,
    donor_ids: list[str],
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    value_col: str,
    scale_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    gap_rows: list[pd.DataFrame] = []
    for placebo_id in donor_ids:
        controls = [d for d in donor_ids if d != placebo_id]
        if len(controls) < 2:
            continue
        try:
            result = run_synth(
                panel,
                placebo_id,
                controls,
                pre_months,
                all_months,
                baseline_month,
                value_col=value_col,
                scale_mode=scale_mode,
            )
        except Exception:
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "id_municipio_6": placebo_id,
                "pre_rmspe_index": metrics["pre_rmspe_index"],
                "pre_mspe_index": metrics["pre_rmspe_index"] ** 2,
                "post_pre_rmspe_ratio": metrics["post_pre_rmspe_ratio"],
                "mean_post_effect_value": metrics["mean_post_effect_value"],
                "last_effect_value": metrics["last_effect_value"],
            }
        )
        ts = result["timeseries"][["anomes", "effect_value", "effect_index"]].copy()
        ts["id_municipio_6"] = placebo_id
        gap_rows.append(ts)
    metrics_df = pd.DataFrame(rows)
    gaps_df = pd.concat(gap_rows, ignore_index=True) if gap_rows else pd.DataFrame()
    return metrics_df, gaps_df


def p_value(treated_ratio: float, placebos: pd.DataFrame) -> float | None:
    if placebos.empty:
        return None
    return float((1 + (placebos["post_pre_rmspe_ratio"] >= treated_ratio).sum()) / (1 + len(placebos)))


def run_analysis() -> tuple[pd.DataFrame, dict]:
    config = load_config()
    treated_id = config["treatment"]["id_municipio_mds"]
    base_first_pre_month = int(config["treatment"].get("first_pre_month", 202303))
    intervention_month = int(config["treatment"]["intervention_month"])
    baseline_month = int(config["treatment"]["last_pre_month"])
    post_end = int(config["treatment"]["post_end_month"])
    fake_baseline_month = 202404

    OUT.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    summary_rows: list[dict] = []
    panels: dict[str, pd.DataFrame] = {}
    baseline_covariates = load_baseline_covariates()

    for outcome_key, spec in config["outcomes"].items():
        matching_spec = OUTCOME_MATCHING_SPECS.get(outcome_key, DEFAULT_MATCHING_SPEC)
        first_pre_month = base_first_pre_month
        for _ in range(int(matching_spec.get("extra_pre_months", 0))):
            first_pre_month = previous_month(first_pre_month, 1)
        pre_months = month_sequence(first_pre_month, baseline_month)
        fake_pre_months = month_sequence(first_pre_month, 202404)
        fake_post_months = month_sequence(202405, baseline_month)
        panel = panels.setdefault(spec["panel"], load_panel(spec["panel"]))
        value_col = spec["value_col"]
        scale_mode = spec.get("scale_mode", "index")
        require_positive = scale_mode == "index"
        outcome_post_end = int(spec.get("post_end_month", post_end))
        all_months = month_sequence(first_pre_month, outcome_post_end)
        complete_units = units_with_complete_panel(
            panel, all_months, value_col=value_col, require_positive=require_positive
        )
        lagged_y_months = matching_lag_months(outcome_key, intervention_month, pre_months)
        features = build_features(
            panel[panel["id_municipio_6"].isin(complete_units)],
            pre_months,
            lagged_y_months=lagged_y_months,
            value_col=value_col,
            auxiliary_col="beneficio_medio" if spec["panel"] == "bolsa" else None,
            require_positive=require_positive,
        )
        labels = panel[["id_municipio_6", "municipio", "sigla_uf"]].drop_duplicates()
        features = features.merge(labels, on=["id_municipio_6", "municipio", "sigla_uf"], how="left")
        if bool(matching_spec.get("baseline_covariates")) and not baseline_covariates.empty:
            features = features.merge(baseline_covariates, on="id_municipio_6", how="left")

        for pool_key, pool_spec in POOLS.items():
            candidate_ids = build_candidate_ids(panel, complete_units, treated_id, pool_spec["ufs"])
            matched, donor_selection = select_donor_pool(
                panel,
                features,
                treated_id,
                candidate_ids,
                pre_months,
                all_months,
                baseline_month,
                value_col,
                scale_mode,
                labels,
            )
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
            placebo_metrics, placebo_gaps = placebo_distribution(
                panel,
                donor_ids,
                pre_months,
                all_months,
                baseline_month,
                value_col,
                scale_mode,
            )
            treated_mspe = result["metrics"]["pre_rmspe_index"] ** 2
            if placebo_metrics.empty:
                good_placebos = placebo_metrics
                good_gaps = placebo_gaps
            else:
                good_ids = set(
                    placebo_metrics.loc[
                        placebo_metrics["pre_mspe_index"] <= 5.0 * treated_mspe,
                        "id_municipio_6",
                    ]
                )
                good_placebos = placebo_metrics[placebo_metrics["id_municipio_6"].isin(good_ids)]
                good_gaps = placebo_gaps[placebo_gaps["id_municipio_6"].isin(good_ids)]
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
            fp_error = ""
            try:
                fp_ci, fp_meta = confidence_set_for_panel(
                    panel,
                    treated_id,
                    donor_ids,
                    pre_months,
                    all_months,
                    baseline_month,
                    value_col,
                    scale_mode,
                    alpha=0.10,
                    precision=30,
                    effect_type="linear",
                    ma_window=3,
                )
                fp_lb_param = fp_meta.lb_param
                fp_ub_param = fp_meta.ub_param
                fp_initial_param = fp_meta.initial_param
                fp_significance = fp_meta.significance
                fp_p_value = fp_meta.zero_p_value
            except RuntimeError as exc:
                fp_error = str(exc)
                fp_ci = pd.DataFrame(
                    {
                        "anomes": all_months,
                        "lower_index": pd.NA,
                        "upper_index": pd.NA,
                        "lower_value": pd.NA,
                        "upper_value": pd.NA,
                        "fp_initial_param": pd.NA,
                        "fp_lb_param": pd.NA,
                        "fp_ub_param": pd.NA,
                        "fp_significance": pd.NA,
                        "fp_effect_type": "linear",
                        "fp_error": fp_error,
                    }
                )
                fp_lb_param = float("nan")
                fp_ub_param = float("nan")
                fp_initial_param = float("nan")
                fp_significance = float("nan")
                fp_p_value = float("nan")
            key = f"{outcome_key}_{pool_key}"
            ts = result["timeseries"].copy()
            ts.to_csv(OUT / f"{key}_timeseries.csv", index=False)
            matched.to_csv(OUT / f"{key}_matched.csv", index=False)
            result["weights"].merge(labels, on="id_municipio_6", how="left").to_csv(
                OUT / f"{key}_weights.csv", index=False
            )
            good_placebos.to_csv(OUT / f"{key}_placebos_goodfit.csv", index=False)
            good_gaps.to_csv(OUT / f"{key}_placebo_gaps_goodfit.csv", index=False)
            loo.to_csv(OUT / f"{key}_leave_one_out.csv", index=False)
            fake["timeseries"].to_csv(OUT / f"{key}_time_placebo.csv", index=False)
            fp_ci.to_csv(OUT / f"{key}_firpo_possebom_ci.csv", index=False)

            metrics = result["metrics"]
            placebo_p_value = p_value(metrics["post_pre_rmspe_ratio"], good_placebos)
            pre_gap = ts.loc[ts["anomes"].isin(pre_months), "effect_value"]
            post_gap = ts.loc[~ts["anomes"].isin(pre_months), "effect_value"]
            row = {
                "outcome": outcome_key,
                "pool": pool_key,
                "matching_spec": matching_spec.get("name", "all_pre_lags"),
                "matching_lags": ",".join(str(month) for month in lagged_y_months)
                if lagged_y_months
                else "none",
                "n_donors": len(donor_ids),
                "donor_selection_order": donor_selection.get("donor_selection_order"),
                "donor_selection_pre_rmspe": donor_selection.get("donor_selection_pre_rmspe"),
                "baseline_covariates": "pop_pib"
                if bool(matching_spec.get("baseline_covariates")) and not baseline_covariates.empty
                else "none",
                "scale_mode": scale_mode,
                "pre_rmspe": metrics["pre_rmspe_index"],
                "post_pre_ratio": metrics["post_pre_rmspe_ratio"],
                "p_value": fp_p_value,
                "fp_p_value": fp_p_value,
                "placebo_p_value": placebo_p_value,
                "mean_post_effect": metrics["mean_post_effect_value"],
                "last_actual": metrics["last_actual_value"],
                "last_synth": metrics["last_synth_value"],
                "last_effect": metrics["last_effect_value"],
                "pre_t_stat": t_stat(pre_gap),
                "post_t_stat": t_stat(post_gap),
                "n_placebos_goodfit": len(good_placebos),
                "fake_mean_effect": fake["metrics"]["mean_post_effect_value"],
                "loo_min": None if loo.empty else loo["last_effect_value"].min(),
                "loo_max": None if loo.empty else loo["last_effect_value"].max(),
                "fp_ci_lb_param": fp_lb_param,
                "fp_ci_ub_param": fp_ub_param,
                "fp_ci_initial_param": fp_initial_param,
                "fp_ci_significance": fp_significance,
                "fp_ci_error": fp_error,
            }
            summary_rows.append(row)
            results[key] = {
                "panel": panel,
                "matched": matched,
                "weights": result["weights"].merge(labels, on="id_municipio_6", how="left"),
                "timeseries": ts,
                "placebos": good_placebos,
                "placebo_gaps": good_gaps,
                "fp_ci": fp_ci,
                "summary": row,
                "label": spec["label"],
                "intervention_month": intervention_month,
                "pre_months": pre_months,
            }

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    return summary, results


def _panel_scale(values: list[float], low: float, high: float, left: float, bottom: float, width: float, height: float):
    denom = max(high - low, 1e-9)

    def xy(i: int, value: float, n: int) -> tuple[float, float]:
        x = left + i * width / max(n - 1, 1)
        y = bottom + (value - low) * height / denom
        return x, y

    return xy


def draw_line(c: canvas.Canvas, xs: list[float], ys: list[float], color, width: float = 1.0, dash: bool = False):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setDash(3, 2) if dash else c.setDash()
    for i in range(len(xs) - 1):
        if pd.notna(ys[i]) and pd.notna(ys[i + 1]):
            c.line(xs[i], ys[i], xs[i + 1], ys[i + 1])
    c.setDash()


def draw_axes(
    c: canvas.Canvas,
    left: float,
    bottom: float,
    width: float,
    height: float,
    months: list[int],
    treatment_month: int,
    y_label: str = "",
    x_label: bool = False,
    y_low: float | None = None,
    y_high: float | None = None,
    y_ticks: list[float] | None = None,
) -> None:
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.45)
    c.line(left, bottom, left + width, bottom)
    c.line(left, bottom, left, bottom + height)
    if treatment_month in months:
        idx = months.index(treatment_month)
        x = left + idx * width / max(len(months) - 1, 1)
        c.setStrokeColor(colors.HexColor("#c6c6c6"))
        c.setDash(5, 5)
        c.line(x, bottom, x, bottom + height)
        c.setDash()
    c.setFont("Helvetica", 6.4)
    c.setFillColor(colors.black)
    tick_positions = [0, len(months) // 2, len(months) - 1]
    for idx in tick_positions:
        x = left + idx * width / max(len(months) - 1, 1)
        label = month_label(months[idx])
        if idx == 0:
            c.drawString(x, bottom - 11, label)
        elif idx == len(months) - 1:
            c.drawRightString(x, bottom - 11, label)
        else:
            c.drawCentredString(x, bottom - 11, label)
    if x_label:
        c.setFont("Helvetica", 7.2)
        c.drawCentredString(left + width / 2, bottom - 31, "Tempo")
    if y_low is not None and y_high is not None and y_high > y_low:
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.35)
        c.setFont("Helvetica", 5.5)
        tick_values = y_ticks if y_ticks is not None else [y_low, (y_low + y_high) / 2, y_high]
        for value in tick_values:
            if value < y_low or value > y_high:
                continue
            y = bottom + (float(value) - y_low) * height / (y_high - y_low)
            c.line(left - 2.5, y, left, y)
            c.drawRightString(left - 4.5, y - 1.8, axis_tick_label(value))
    if y_label:
        c.saveState()
        c.translate(left - 39, bottom + height / 2)
        c.rotate(90)
        c.setFont("Helvetica", 6.3)
        c.drawCentredString(0, 0, y_label)
        c.restoreState()


def draw_actual_synth_panel(
    c: canvas.Canvas,
    area: tuple[float, float, float, float],
    ts: pd.DataFrame,
    title: str,
    treatment_month: int,
    ma: int | None,
    y_label: str = "",
) -> None:
    left, bottom, width, height = area
    months = ts["anomes"].astype(int).tolist()
    actual = ts["actual_value"].astype(float)
    synth = ts["synth_value"].astype(float)
    if ma:
        actual = moving_average(actual, ma)
        synth = moving_average(synth, ma)
    values = actual.tolist() + synth.tolist()
    low, high = min(values), max(values)
    pad = (high - low) * 0.08 if high > low else 1
    y_low, y_high = low - pad, high + pad
    xy = _panel_scale(values, y_low, y_high, left, bottom, width, height)
    xs = [xy(i, v, len(months))[0] for i, v in enumerate(actual)]
    y_actual = [xy(i, v, len(months))[1] for i, v in enumerate(actual)]
    y_synth = [xy(i, v, len(months))[1] for i, v in enumerate(synth)]
    draw_axes(
        c,
        left,
        bottom,
        width,
        height,
        months,
        treatment_month,
        y_label=y_label,
        y_low=y_low,
        y_high=y_high,
        y_ticks=rounded_axis_ticks(y_low, y_high, minimum_step=100),
    )
    draw_line(c, xs, y_actual, colors.black, width=1.1)
    draw_line(c, xs, y_synth, colors.black, width=0.75, dash=True)
    c.setFillColor(colors.black)
    if title:
        c.setFont("Times-Roman", 8.3)
        c.drawCentredString(left + width / 2, bottom + height + 12, title)


def draw_placebo_panel(
    c: canvas.Canvas,
    area: tuple[float, float, float, float],
    ts: pd.DataFrame,
    placebo_gaps: pd.DataFrame,
    title: str,
    treatment_month: int,
    ma: int | None,
    y_label: str = "",
) -> None:
    left, bottom, width, height = area
    months = ts["anomes"].astype(int).tolist()
    treated = ts["effect_value"].astype(float)
    if ma:
        treated = moving_average(treated, ma)
    placebo_series: list[pd.Series] = []
    values = treated.tolist()
    if not placebo_gaps.empty:
        for _, group in placebo_gaps.sort_values("anomes").groupby("id_municipio_6"):
            s = group.set_index("anomes").reindex(months)["effect_value"].astype(float)
            if ma:
                s = moving_average(s, ma)
            placebo_series.append(s)
            values.extend(s.dropna().tolist())
    low, high = min(values + [0.0]), max(values + [0.0])
    pad = (high - low) * 0.08 if high > low else 1
    y_low, y_high = low - pad, high + pad
    xy = _panel_scale(values, y_low, y_high, left, bottom, width, height)
    draw_axes(
        c,
        left,
        bottom,
        width,
        height,
        months,
        treatment_month,
        y_label=y_label,
        y_low=y_low,
        y_high=y_high,
        y_ticks=rounded_axis_ticks(
            y_low, y_high, include_zero=True, minimum_step=100, target_ticks=6
        ),
    )
    zero_y = xy(0, 0.0, len(months))[1]
    c.setStrokeColor(colors.HexColor("#c8c8c8"))
    c.setDash(5, 4)
    c.line(left, zero_y, left + width, zero_y)
    c.setDash()
    for s in placebo_series:
        xs = [xy(i, v, len(months))[0] for i, v in enumerate(s)]
        ys = [xy(i, v, len(months))[1] for i, v in enumerate(s)]
        draw_line(c, xs, ys, colors.HexColor("#c7c7c7"), width=0.35)
    xs = [xy(i, v, len(months))[0] for i, v in enumerate(treated)]
    ys = [xy(i, v, len(months))[1] for i, v in enumerate(treated)]
    draw_line(c, xs, ys, colors.black, width=1.1)
    c.setFillColor(colors.black)
    if title:
        c.setFont("Times-Roman", 8.3)
        c.drawCentredString(left + width / 2, bottom + height + 12, title)


def draw_ci_panel(
    c: canvas.Canvas,
    area: tuple[float, float, float, float],
    ts: pd.DataFrame,
    placebo_gaps: pd.DataFrame,
    title: str,
    treatment_month: int,
    ma: int | None,
    fp_ci: pd.DataFrame | None = None,
    y_label: str = "",
) -> None:
    left, bottom, width, height = area
    months = ts["anomes"].astype(int).tolist()
    treated = ts["effect_value"].astype(float)
    if ma:
        treated = moving_average(treated, ma)
    use_fp_ci = (
        fp_ci is not None
        and not fp_ci.empty
        and fp_ci[["lower_value", "upper_value"]].notna().any().all()
    )
    if use_fp_ci:
        ci = fp_ci.set_index("anomes").reindex(months)
        lower = ci["lower_value"].astype(float).reset_index(drop=True)
        upper = ci["upper_value"].astype(float).reset_index(drop=True)
    else:
        lower = treated.reset_index(drop=True)
        upper = treated.reset_index(drop=True)
    values = treated.tolist() + lower.tolist() + upper.tolist() + [0.0]
    low, high = min(values), max(values)
    pad = (high - low) * 0.08 if high > low else 1
    y_low, y_high = low - pad, high + pad
    xy = _panel_scale(values, y_low, y_high, left, bottom, width, height)
    draw_axes(
        c,
        left,
        bottom,
        width,
        height,
        months,
        treatment_month,
        y_label=y_label,
        y_low=y_low,
        y_high=y_high,
        y_ticks=rounded_axis_ticks(
            y_low, y_high, include_zero=True, minimum_step=100, target_ticks=6
        ),
    )
    zero_y = xy(0, 0.0, len(months))[1]
    c.setStrokeColor(colors.HexColor("#c8c8c8"))
    c.setDash(5, 4)
    c.line(left, zero_y, left + width, zero_y)
    c.setDash()
    if treatment_month in months and use_fp_ci:
        start = months.index(treatment_month)
        path = c.beginPath()
        for i in range(start, len(months)):
            x, y = xy(i, upper.iloc[i], len(months))
            if i == start:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        for i in range(len(months) - 1, start - 1, -1):
            x, y = xy(i, lower.iloc[i], len(months))
            path.lineTo(x, y)
        path.close()
        c.setFillColor(colors.HexColor("#eeeeee"))
        c.setStrokeColor(colors.HexColor("#dedede"))
        c.drawPath(path, fill=1, stroke=1)
    xs = [xy(i, v, len(months))[0] for i, v in enumerate(treated)]
    ys = [xy(i, v, len(months))[1] for i, v in enumerate(treated)]
    draw_line(c, xs, ys, colors.black, width=1.1)
    c.setFillColor(colors.black)
    if title:
        c.setFont("Times-Roman", 8.3)
        c.drawCentredString(left + width / 2, bottom + height + 12, title)


def draw_pool_comparison_figure(
    rs_result: dict,
    south_result: dict,
    path: Path,
    outcome_short: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = 7.2 * inch, 6.05 * inch
    c = canvas.Canvas(str(path), pagesize=(page_w, page_h))
    margin_x, margin_y = 0.45 * inch, 0.58 * inch
    gutter_x, gutter_y = 0.32 * inch, 0.62 * inch
    panel_w = (page_w - 2 * margin_x - 2 * gutter_x) / 3
    panel_h = (page_h - 2 * margin_y - gutter_y) / 2
    top = margin_y + panel_h + gutter_y
    areas = [
        (margin_x + j * (panel_w + gutter_x), top, panel_w, panel_h)
        for j in range(3)
    ] + [
        (margin_x + j * (panel_w + gutter_x), margin_y, panel_w, panel_h)
        for j in range(3)
    ]
    for offset, pool_label, result in [
        (0, "RS", rs_result),
        (3, "South", south_result),
    ]:
        ts = result["timeseries"]
        gaps = result["placebo_gaps"]
        tr = result["intervention_month"]
        draw_actual_synth_panel(
            c,
            areas[offset],
            ts,
            f"({chr(97 + offset)}) {pool_label}: actual and synthetic",
            tr,
            ma=None,
        )
        draw_placebo_panel(
            c,
            areas[offset + 1],
            ts,
            gaps,
            f"({chr(98 + offset)}) {pool_label}: placebo gaps",
            tr,
            ma=None,
        )
        draw_ci_panel(
            c,
            areas[offset + 2],
            ts,
            gaps,
            f"({chr(99 + offset)}) {pool_label}: gap and C.I.",
            tr,
            ma=None,
        )
    c.setFont("Times-Roman", 9)
    c.drawCentredString(page_w / 2, page_h - 0.22 * inch, outcome_short)
    c.setFont("Helvetica", 6.8)
    y = 0.16 * inch
    x = 1.55 * inch
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.2)
    c.line(x, y, x + 0.32 * inch, y)
    c.drawString(x + 0.37 * inch, y - 2, "Bento Gonçalves/GAP")
    x += 1.15 * inch
    c.setDash(5, 4)
    c.line(x, y, x + 0.32 * inch, y)
    c.setDash()
    c.drawString(x + 0.37 * inch, y - 2, "Synthetic")
    x += 1.05 * inch
    c.setFillColor(colors.HexColor("#eeeeee"))
    c.rect(x, y - 4, 0.30 * inch, 8, fill=1, stroke=1)
    c.setFillColor(colors.black)
    c.drawString(x + 0.35 * inch, y - 2, "C.I.")
    c.save()


def draw_single_pool_figure(result: dict, path: Path, outcome_short: str, pool_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = 7.2 * inch, 3.25 * inch
    c = canvas.Canvas(str(path), pagesize=(page_w, page_h))
    margin_x, margin_y = 0.45 * inch, 0.50 * inch
    gutter_x = 0.32 * inch
    panel_w = (page_w - 2 * margin_x - 2 * gutter_x) / 3
    panel_h = page_h - 1.16 * inch
    areas = [
        (margin_x + j * (panel_w + gutter_x), margin_y + 0.16 * inch, panel_w, panel_h)
        for j in range(3)
    ]
    ts = result["timeseries"]
    gaps = result["placebo_gaps"]
    tr = result["intervention_month"]
    draw_actual_synth_panel(
        c,
        areas[0],
        ts,
        "(a) Actual and synthetic",
        tr,
        ma=None,
    )
    draw_placebo_panel(
        c,
        areas[1],
        ts,
        gaps,
        "(b) Placebo gaps",
        tr,
        ma=None,
    )
    draw_ci_panel(
        c,
        areas[2],
        ts,
        gaps,
        "(c) Gap and C.I.",
        tr,
        ma=None,
    )
    c.setFont("Times-Roman", 9)
    c.drawCentredString(page_w / 2, page_h - 0.18 * inch, f"{outcome_short}: {pool_label}")
    c.setFont("Helvetica", 6.8)
    y = 0.14 * inch
    x = 1.55 * inch
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.2)
    c.line(x, y, x + 0.32 * inch, y)
    c.drawString(x + 0.37 * inch, y - 2, "Bento Gonçalves/GAP")
    x += 1.15 * inch
    c.setDash(5, 4)
    c.line(x, y, x + 0.32 * inch, y)
    c.setDash()
    c.drawString(x + 0.37 * inch, y - 2, "Synthetic")
    x += 1.05 * inch
    c.setFillColor(colors.HexColor("#eeeeee"))
    c.rect(x, y - 4, 0.30 * inch, 8, fill=1, stroke=1)
    c.setFillColor(colors.black)
    c.drawString(x + 0.35 * inch, y - 2, "C.I.")
    c.save()


def draw_panel_pdf(result: dict, path: Path, panel: str, outcome_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = 3.10 * inch, 1.48 * inch
    c = canvas.Canvas(str(path), pagesize=(page_w, page_h))
    area = (0.68 * inch, 0.25 * inch, page_w - 0.83 * inch, page_h - 0.33 * inch)
    ts = result["timeseries"]
    gaps = result["placebo_gaps"]
    tr = result["intervention_month"]
    if panel == "series":
        draw_actual_synth_panel(
            c, area, ts, "", tr, ma=3, y_label=Y_AXIS_LABELS.get(outcome_key, "")
        )
    elif panel == "placebo":
        draw_placebo_panel(
            c, area, ts, gaps, "", tr, ma=3, y_label=EFFECT_AXIS_LABELS.get(outcome_key, "Gap")
        )
    elif panel == "ci":
        draw_ci_panel(
            c,
            area,
            ts,
            gaps,
            "",
            tr,
            ma=3,
            fp_ci=result.get("fp_ci"),
            y_label=EFFECT_AXIS_LABELS.get(outcome_key, "ATT"),
        )
    else:
        raise ValueError(panel)
    c.save()


def weights_note(weights: pd.DataFrame, max_items: int = 5) -> str:
    top = weights.sort_values("weight", ascending=False).head(max_items)
    parts = []
    for _, row in top.iterrows():
        if row["weight"] <= 0:
            continue
        parts.append(f"{row.get('municipio', row['id_municipio_6'])} = {row['weight']:.3f}")
    return "; ".join(parts)


def weights_composition_note(weights: pd.DataFrame, label: str, max_items: int = 3) -> str:
    positive = weights[pd.to_numeric(weights["weight"], errors="coerce") > 0.001].copy()
    top = positive.sort_values("weight", ascending=False).head(max_items)
    parts = []
    for _, row in top.iterrows():
        municipio = str(row.get("municipio", row["id_municipio_6"]))
        uf = str(row.get("sigla_uf", ""))
        weight = 100.0 * float(row["weight"])
        parts.append(f"{municipio}/{uf} {weight:.1f}%")
    count = len(positive)
    detail = ", ".join(parts)
    return f"{label} tem {count} municípios com peso positivo, liderados por {detail}"


def significance_stars(p_value: float | int | None) -> str:
    if p_value is None or pd.isna(p_value):
        return ""
    p = float(p_value)
    if p < 0.01:
        return r"\textsuperscript{***}"
    if p < 0.05:
        return r"\textsuperscript{**}"
    if p < 0.10:
        return r"\textsuperscript{*}"
    return ""


def write_tables(summary: pd.DataFrame, results: dict) -> None:
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    main_outcomes = MAIN_OUTCOMES
    appendix_extra_outcomes = APPENDIX_EXTRA_OUTCOMES
    order = main_outcomes + appendix_extra_outcomes

    def table_rows(specs: list[tuple[str, str]]) -> str:
        rows = []
        previous_outcome = None
        for outcome_key, pool_key in specs:
            row = summary[(summary["outcome"] == outcome_key) & (summary["pool"] == pool_key)].iloc[0]
            outcome_label = (
                OUTCOME_TABLE_LABELS.get(row["outcome"], OUTCOME_LABELS.get(row["outcome"], row["outcome"]))
                if row["outcome"] != previous_outcome
                else ""
            )
            rows.append(
                " & ".join(
                    [
                        tex_escape(outcome_label),
                        tex_escape("RS" if row["pool"] == "rs" else "Sul"),
                        fmt0(row["n_donors"]),
                        fmt0(row["n_placebos_goodfit"]),
                        fmt(row["pre_rmspe"], 2),
                        fmt(row["post_pre_ratio"], 2),
                        fmt(row["p_value"], 3),
                        fmt_effect(row["last_effect"], str(row["outcome"]))
                        + significance_stars(row["p_value"]),
                    ]
                )
                + r" \\"
            )
            previous_outcome = row["outcome"]
        return "\n".join(rows)

    def table_tex(rows: str, note: str) -> str:
        return r"""\begin{threeparttable}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l c r r r r r r@{}}
\toprule
Desfecho & Pool & Doadores & Placebos & RMSPE pré & Pós/pré & $p_{FP}$ & Efeito final \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular*}
\tabnotes{\textit{Notas:} """ + note + r"""}
\end{threeparttable}
"""

    def appendix_table_rows(specs: list[tuple[str, str]]) -> str:
        rows = []
        for outcome_key, pool_key in specs:
            row = summary[(summary["outcome"] == outcome_key) & (summary["pool"] == pool_key)].iloc[0]
            rows.append(
                " & ".join(
                    [
                        tex_escape(OUTCOME_SHORT_LABELS.get(row["outcome"], row["outcome"])),
                        tex_escape("RS" if row["pool"] == "rs" else "Sul"),
                        fmt(row["post_pre_ratio"], 2),
                        fmt(row["p_value"], 3),
                        fmt_effect(row["last_effect"], str(row["outcome"]))
                        + significance_stars(row["p_value"]),
                    ]
                )
                + r" \\"
            )
        return "\n".join(rows)

    def appendix_table_tex(rows: str, note: str) -> str:
        return r"""\begin{threeparttable}
\scriptsize
\begin{tabular}{l c r r r}
\toprule
Desfecho & Pool & Pós/pré & $p_{FP}$ & Efeito final \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\tabnotes{\textit{Notas:} """ + note + r"""}
\end{threeparttable}
"""

    main_specs = [(outcome_key, pool_key) for outcome_key in main_outcomes for pool_key in ["south", "rs"]]
    appendix_specs = [(outcome_key, "south") for outcome_key in appendix_extra_outcomes] + [
        (outcome_key, "rs") for outcome_key in main_outcomes
    ]
    main_weight_note = tex_escape(
        ". ".join(
            weights_composition_note(
                results[f"{outcome_key}_south"]["weights"],
                OUTCOME_SHORT_LABELS.get(outcome_key, outcome_key),
                max_items=3,
            )
            for outcome_key in main_outcomes
        )
        + "."
    )

    main_note = (
        "A tabela reporta os desfechos destacados no texto principal, em suas unidades indicadas. "
        "Doadores é o número de municípios retidos pela rotina de ajuste pré-tratamento. "
        "Placebos é o número de doadores reestimados como tratados e mantidos no gráfico por terem MSPE pré até cinco vezes o MSPE pré do município de Bento Gonçalves. "
        "Os $p$-valores reportam o teste de efeito nulo da rotina SCM.CS de Firpo e Possebom (2018). "
        "Estrelas no efeito final indicam \\textsuperscript{*}$p<0{,}10$, "
        "\\textsuperscript{**}$p<0{,}05$ e \\textsuperscript{***}$p<0{,}01$. "
        "Pesos principais do pool sintético: "
        + main_weight_note
    )
    appendix_note = (
        "A tabela reporta a robustez dos desfechos principais com pool restrito ao Rio Grande do Sul. "
        "Aplica-se o mesmo filtro de qualidade de ajuste pré-intervenção usado na tabela principal."
    )
    (PAPER / "tables" / "tab_results.tex").write_text(
        table_tex(table_rows(main_specs), main_note),
        encoding="utf-8",
    )
    (PAPER / "tables" / "tab_appendix_results.tex").write_text(
        appendix_table_tex(appendix_table_rows(appendix_specs), appendix_note),
        encoding="utf-8",
    )

    # Keep a stable order in the exported CSV summary for quick manual inspection.
    summary_ordered = summary.copy()
    summary_ordered["outcome_order"] = summary_ordered["outcome"].map({k: i for i, k in enumerate(order)})
    summary_ordered.sort_values(["outcome_order", "pool"]).to_csv(OUT / "summary_ordered.csv", index=False)

    notes = {
        "bolsa": weights_note(results["bolsa_familia_south"]["weights"]),
        "stock": weights_note(results["emprego_estoque_south"]["weights"]),
    }
    (PAPER / "tables" / "fig_notes.tex").write_text(
        "\\newcommand{\\BolsaWeights}{"
        + tex_escape(notes["bolsa"])
        + "}\n\\newcommand{\\StockWeights}{"
        + tex_escape(notes["stock"])
        + "}\n",
        encoding="utf-8",
    )


def load_rais_bento() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "rais_income_municipio_panel.csv"
    rais = pd.read_csv(path, dtype={"id_municipio": str})
    value_cols = [
        "vinculos_ativos_3112",
        "remuneracao_media",
        "remuneracao_dezembro_media",
        "horas_contratadas_media",
    ]
    for col in value_cols:
        rais[col] = pd.to_numeric(rais[col], errors="coerce")
    return rais[rais["id_municipio"] == "4302105"].sort_values("ano").copy()


def write_rais_table() -> None:
    rais = load_rais_bento()
    rows = []
    for _, row in rais.iterrows():
        rows.append(
            " & ".join(
                [
                    str(int(row["ano"])),
                    fmt0(row["vinculos_ativos_3112"]),
                    fmt0(row["remuneracao_media"]),
                    fmt0(row["remuneracao_dezembro_media"]),
                    fmt(row["horas_contratadas_media"], 1),
                ]
            )
            + r" \\"
        )
    content = (
        r"""\begin{threeparttable}
\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{l r r r r}
\toprule
Ano & Vínculos 31/12 & Remun. média (R\$) & Remun. dez. (R\$) & Horas \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\tabnotes{\textit{Notas:} RAIS/Base dos Dados, município de Bento Gonçalves. Valores nominais.}
\end{threeparttable}
"""
    )
    (PAPER / "tables" / "tab_rais_bento.tex").write_text(content, encoding="utf-8")


def write_references() -> None:
    content = r"""@article{abadie2010synthetic,
  author = {Abadie, Alberto and Diamond, Alexis and Hainmueller, Jens},
  title = {Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program},
  journal = {Journal of the American Statistical Association},
  year = {2010},
  volume = {105},
  number = {490},
  pages = {493--505}
}

@article{firpo2018synthetic,
  author = {Firpo, Sergio and Possebom, Vitor},
  title = {Synthetic Control Method: Inference, Sensitivity Analysis and Confidence Sets},
  journal = {Journal of Causal Inference},
  year = {2018},
  volume = {6},
  number = {2}
}

@article{ferman2020cherrypicking,
  author = {Ferman, Bruno and Pinto, Cristine and Possebom, Vitor},
  title = {Cherry Picking with Synthetic Controls},
  journal = {Journal of Policy Analysis and Management},
  year = {2020},
  volume = {39},
  number = {2},
  pages = {510--532},
  doi = {10.1002/pam.22206}
}

@article{card2010active,
  author = {Card, David and Kluve, Jochen and Weber, Andrea},
  title = {Active Labour Market Policy Evaluations: A Meta-Analysis},
  journal = {The Economic Journal},
  year = {2010},
  volume = {120},
  number = {548},
  pages = {F452--F477},
}

@article{card2018what,
  author = {Card, David and Kluve, Jochen and Weber, Andrea},
  title = {What Works? A Meta Analysis of Recent Active Labor Market Program Evaluations},
  journal = {Journal of the European Economic Association},
  year = {2018},
  volume = {16},
  number = {3},
  pages = {894--931},
}

@article{crepon2016active,
  author = {Crepon, Bruno and van den Berg, Gerard J.},
  title = {Active Labor Market Policies},
  journal = {Annual Review of Economics},
  year = {2016},
  volume = {8},
  pages = {521--546},
}

@article{glewwe2012bolsa,
  author = {Glewwe, Paul and Kassouf, Ana Lucia},
  title = {The Impact of the Bolsa Escola/Familia Conditional Cash Transfer Program on Enrollment, Dropout Rates and Grade Promotion in Brazil},
  journal = {Journal of Development Economics},
  year = {2012},
  volume = {97},
  number = {2},
  pages = {505--517},
}

@article{debrauw2015labor,
  author = {de Brauw, Alan and Gilligan, Daniel O. and Hoddinott, John and Roy, Shalini},
  title = {Bolsa Familia and Household Labor Supply},
  journal = {Economic Development and Cultural Change},
  year = {2015},
  volume = {63},
  number = {3},
  pages = {423--457},
}

@article{debrauw2015schooling,
  author = {de Brauw, Alan and Gilligan, Daniel O. and Hoddinott, John and Roy, Shalini},
  title = {The Impact of Bolsa Familia on Schooling},
  journal = {World Development},
  year = {2015},
  volume = {70},
  pages = {303--316},
}

@article{barbosa2014informality,
  author = {Barbosa, Ana Luiza Neves de Holanda and Corseuil, Carlos Henrique Leite},
  title = {Conditional Cash Transfer and Informality in Brazil},
  journal = {IZA Journal of Labor and Development},
  year = {2014},
  volume = {3},
  number = {37},
}

@article{santos2017duration,
  author = {Santos, Danilo Braun and Leichsenring, Alexandre Ribeiro and Menezes Filho, Naercio and Mendes-Da-Silva, Wesley},
  title = {The Impact of the Bolsa Familia Program on the Duration of Formal Employment for Low Income Individuals},
  journal = {Revista de Administracao Publica},
  year = {2017},
  volume = {51},
  number = {5},
  pages = {708--733},
}

@article{fassarella2024mobility,
  author = {Fassarella, Eloah and Ferreira, Sergio and Franco, Samuel and Franco, Stefano and Pinho Neto, Valdemar and Ribeiro, Giovanna and Schuabb, Vinicius and Tafner, Paulo},
  title = {Social Mobility and CCT Programs: The Bolsa Familia Program in Brazil},
  journal = {World Development Perspectives},
  year = {2024},
  volume = {35},
  pages = {100624},
  doi = {10.1016/j.wdp.2024.100624},
}

@article{banerjee2017lazy,
  author = {Banerjee, Abhijit V. and Hanna, Rema and Kreindler, Gabriel E. and Olken, Benjamin A.},
  title = {Debunking the Stereotype of the Lazy Welfare Recipient: Evidence from Cash Transfer Programs},
  journal = {The World Bank Research Observer},
  year = {2017},
  volume = {32},
  number = {2},
  pages = {155--184},
}

@article{molinamillan2019longterm,
  author = {Molina Millan, Teresa and Barham, Tania and Macours, Karen and Maluccio, John A. and Stampini, Marco},
  title = {Long-Term Impacts of Conditional Cash Transfers: Review of the Evidence},
  journal = {The World Bank Research Observer},
  year = {2019},
  volume = {34},
  number = {1},
  pages = {119--159},
}

@article{parker2023cct,
  author = {Parker, Susan W. and Vogl, Tom},
  title = {Do Conditional Cash Transfers Improve Economic Outcomes in the Next Generation? Evidence from Mexico},
  journal = {The Economic Journal},
  year = {2023},
  volume = {133},
  number = {655},
  pages = {2775--2806},
}

@techreport{best2026productive,
  author = {Best, Michael C. and Lobel, Felipe and Pinho Neto, Valdemar},
  title = {Cash Transfers and Productive Inclusion: Evidence from Bolsa Familia},
  institution = {National Bureau of Economic Research},
  type = {Working Paper},
  number = {35006},
  year = {2026},
}

@techreport{pradofirpo2026integridade,
  author = {Prado, Thaline and Firpo, Sergio Pinheiro},
  title = {Propostas infralegais para integridade e correção de benefícios de proteção social: eficiência do gasto e recomposição da capacidade fiscal},
  institution = {Observatório da Qualidade do Gasto Público, Centro de Gestão de Políticas Públicas, Insper},
  type = {Relatório de pesquisa},
  year = {2026},
  url = {https://repositorio.insper.edu.br/handle/11224/8307},
}

@misc{prefeiturabento2025bolsafamilia,
  author = {{Prefeitura Municipal de Bento Gonçalves}},
  title = {Ações para redução do {Bolsa Família} em {Bento Gonçalves}},
  year = {2025},
  note = {Publicado em 8 de outubro de 2025},
  howpublished = {\href{https://www.bentogoncalves.rs.gov.br/reuniao-entre-os-municipios-da-regiao-verifica-andamento-das-acoes-para-reducao-do-bolsa-familia/}{Portal da Prefeitura Municipal de Bento Gonçalves}},
}

@misc{mdsvisdata,
  author = {{Ministerio do Desenvolvimento e Assistencia Social}},
  title = {MDS/VISDATA: municipal series for Bolsa Familia},
  year = {2026},
  howpublished = {\url{https://aplicacoes.mds.gov.br/sagi/servicos/misocial}}
}

@misc{mtecaged2026,
  author = {{Ministerio do Trabalho e Emprego}},
  title = {Novo CAGED: March 2026 municipal tables},
  year = {2026},
  howpublished = {\url{https://www.gov.br/trabalho-e-emprego/pt-br/assuntos/estatisticas-trabalho/novo-caged/2026/marco/pagina-inicial}}
}
"""
    (PAPER / "references.bib").write_text(content, encoding="utf-8")


def write_main_tex(summary: pd.DataFrame, results: dict) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    bolsa = summary[(summary["outcome"] == "bolsa_familia") & (summary["pool"] == "south")].iloc[0]
    stock = summary[(summary["outcome"] == "emprego_estoque") & (summary["pool"] == "south")].iloc[0]
    bolsa_last_pct = 100 * abs(float(bolsa["last_effect"])) / float(bolsa["last_synth"])
    stock_last_pct = 100 * float(stock["last_effect"]) / float(stock["last_synth"])
    last_ratio = float(stock["last_effect"]) / abs(float(bolsa["last_effect"]))
    mean_ratio = float(stock["mean_post_effect"]) / abs(float(bolsa["mean_post_effect"]))

    def fmt_pt(value: float | int, digits: int = 1) -> str:
        return fmt(value, digits).replace(".", ",")

    rais = load_rais_bento()
    rais_2023 = rais[rais["ano"] == 2023].iloc[0]
    rais_2024 = rais[rais["ano"] == 2024].iloc[0]

    def preamble(title: str) -> str:
        return rf"""\documentclass[10pt,twocolumn]{{article}}

\usepackage[letterpaper,margin=0.56in]{{geometry}}
\usepackage{{amsmath,amssymb,graphicx,booktabs,threeparttable,natbib,float,subcaption,placeins,titlesec}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{caption}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=black, citecolor=blue, urlcolor=black}}
\bibliographystyle{{plainnat}}
\setcitestyle{{authoryear,round}}
\captionsetup{{font=scriptsize,labelfont=bf}}
\captionsetup[subfigure]{{font=scriptsize,skip=1pt}}
\titlespacing*{{\section}}{{0pt}}{{6pt plus 1pt minus 1pt}}{{3pt plus 1pt minus 1pt}}
\setlength{{\floatsep}}{{4pt plus 1pt minus 1pt}}
\setlength{{\textfloatsep}}{{5pt plus 1pt minus 1pt}}
\setlength{{\dblfloatsep}}{{4pt plus 1pt minus 1pt}}
\setlength{{\dbltextfloatsep}}{{5pt plus 1pt minus 1pt}}
\setcounter{{topnumber}}{{4}}
\setcounter{{dbltopnumber}}{{4}}
\renewcommand{{\topfraction}}{{0.95}}
\renewcommand{{\dbltopfraction}}{{0.95}}
\renewcommand{{\textfraction}}{{0.03}}
\renewcommand{{\floatpagefraction}}{{0.85}}
\renewcommand{{\dblfloatpagefraction}}{{0.85}}
\setlength{{\parskip}}{{1pt}}
\setlength{{\parindent}}{{10pt}}
\raggedbottom
\renewcommand{{\abstractname}}{{Resumo}}
\renewcommand{{\refname}}{{Referências}}
\renewcommand{{\tablename}}{{Tabela}}
\renewcommand{{\figurename}}{{Figura}}
\newcommand{{\tabnotes}}[1]{{\par\vspace{{2pt}}\noindent\begin{{minipage}}{{\linewidth}}\scriptsize #1\end{{minipage}}}}
\title{{\vspace{{-1.35cm}}\textbf{{{title}}}}}
\author{{Victor Rangel\textsuperscript{{*}}\\[-1pt]{{\normalsize Insper}}}}
\date{{}}
"""

    def outcome_effect_note(outcome_key: str, pool_key: str) -> str:
        row = summary[(summary["outcome"] == outcome_key) & (summary["pool"] == pool_key)].iloc[0]
        if outcome_key == "bolsa_familia":
            return (
                f"{fmt_effect(row['last_effect'], outcome_key)} famílias no Bolsa Família "
                f"($p_{{FP}}={fmt(row['p_value'], 3)}$)"
            )
        if outcome_key == "emprego_estoque":
            return (
                f"{fmt_effect(row['last_effect'], outcome_key)} vínculos no estoque formal "
                f"($p_{{FP}}={fmt(row['p_value'], 3)}$)"
            )
        return (
            f"{fmt_effect(row['last_effect'], outcome_key)} em "
            f"{OUTCOME_SHORT_LABELS.get(outcome_key, outcome_key)} ($p_{{FP}}={fmt(row['p_value'], 3)}$)"
        )

    def pool_composition_note(outcomes: list[str], pool_key: str) -> str:
        notes = []
        for outcome_key in outcomes:
            key = f"{outcome_key}_{pool_key}"
            notes.append(
                weights_composition_note(
                    results[key]["weights"],
                    OUTCOME_SHORT_LABELS.get(outcome_key, outcome_key),
                    max_items=3,
                )
            )
        return tex_escape(". ".join(notes)) + "."

    def panel_wall_figure(
        outcomes: list[str],
        prefix: str,
        pool_label: str,
        label_prefix: str,
        caption_extra: str = "",
        composition_note: str = "",
    ) -> str:
        if len(outcomes) != 2:
            raise ValueError("panel_wall_figure expects exactly two outcomes.")
        panel_titles = {
            "series": "Observado e sintético",
            "ci": "Gap e IC 90\\%",
            "placebo": "Placebos",
        }
        boxes = {}
        for idx, outcome_key in enumerate(outcomes, start=1):
            short = tex_escape(OUTCOME_SHORT_LABELS.get(outcome_key, outcome_key))
            for panel_name in ["series", "ci", "placebo"]:
                basename = f"fig_{label_prefix}_{idx:02d}_{outcome_key}_{panel_name}"
                title = f"{short}: {panel_titles[panel_name]}"
                boxes[(outcome_key, panel_name)] = (
                    rf"\subcaptionbox{{{title}}}[0.486\textwidth]"
                    rf"{{\includegraphics[width=0.486\textwidth]{{figures/{basename}.pdf}}}}"
                )
        rows = []
        for panel_name in ["series", "ci", "placebo"]:
            rows.append(
                boxes[(outcomes[0], panel_name)]
                + r"\hspace{0.006\textwidth}%"
                + "\n"
                + boxes[(outcomes[1], panel_name)]
            )
        rows = ("\n\n" + r"\vspace{2pt}" + "\n").join(rows)
        notes = [outcome_effect_note(outcome_key, prefix) for outcome_key in outcomes]
        if len(notes) == 2:
            effects_sentence = f"No último mês, o efeito é {notes[0]} e {notes[1]}."
        else:
            effects_sentence = "No último mês, o efeito é " + ", ".join(notes) + "."
        extra = f" {caption_extra}" if caption_extra else ""
        composition_block = (
            rf"""
\par\vspace{{1pt}}
\begin{{minipage}}{{0.96\textwidth}}
\footnotesize\emph{{Principais pesos do pool sintético. {composition_note}}}
\end{{minipage}}
"""
            if composition_note
            else ""
        )
        return rf"""\begin{{figure*}}[t]
\centering
\makebox[\textwidth][c]{{%
\begin{{minipage}}{{0.995\textwidth}}
\centering
{rows}
\end{{minipage}}%
}}
\caption{{Controle sintético, pool {pool_label}. Painéis à esquerda reportam famílias beneficiárias. Painéis à direita reportam estoque de vínculos formais. Séries em média móvel de 3 meses. {effects_sentence}{extra}}}
\label{{fig:{label_prefix}-att}}
{composition_block}
\end{{figure*}}
"""

    main_figures = panel_wall_figure(
        MAIN_OUTCOMES,
        "south",
        "Sul",
        "main",
    )
    appendix_figures = panel_wall_figure(
        MAIN_OUTCOMES,
        "rs",
        "Rio Grande do Sul",
        "app-rs",
        "Robustez para o pool restrito ao RS.",
    )

    main_content = (
        preamble("Nota de Política Pública: saídas do Bolsa Família e emprego formal no município de Bento Gonçalves")
        + rf"""
\begin{{document}}
\twocolumn[
\maketitle
\vspace{{-0.55cm}}
\begin{{abstract}}
\noindent Em novembro de 2024, o município de Bento Gonçalves iniciou uma política que combinou revisão cadastral do Bolsa Família, busca ativa de famílias em idade produtiva e encaminhamento a vagas formais. Esta nota estima seu efeito agregado com controle sintético. Em março de 2026, o município tinha {fmt0(abs(bolsa['last_effect']))} famílias a menos no programa do que seu contrafactual, e o estoque formal de empregos estava {fmt0(stock['last_effect'])} vínculos acima da série sintética. Os resultados indicam que a queda de beneficiários veio acompanhada de melhora no mercado formal local, um padrão consistente com inclusão produtiva e relevante para decisões de escala e monitoramento.
\end{{abstract}}
\vspace{{0.05cm}}
\noindent{{\footnotesize \textbf{{Palavras-chave:}} Bolsa Família; controle sintético; política municipal; emprego formal. \textbf{{JEL:}} I38; J68; C23; H53.}}
\vspace{{0.25cm}}
]
\begingroup
\renewcommand{{\thefootnote}}{{*}}
\footnotetext{{E-mail: \href{{mailto:victorrsr@al.insper.edu.br}}{{victorrsr@al.insper.edu.br}}. ORCID: \href{{https://orcid.org/0000-0002-4520-2795}}{{https://orcid.org/0000-0002-4520-2795}}.}}
\endgroup

\section{{Introdução}}

Em novembro de 2024, o município de Bento Gonçalves passou a revisar o cadastro do Bolsa Família e encaminhar famílias em idade produtiva a vagas formais, em ação oficial de busca ativa, checagem cadastral e oferta de emprego \citep{{prefeiturabento2025bolsafamilia}}. A pergunta é se a queda subsequente de beneficiários excedeu a de municípios semelhantes e veio acompanhada de emprego formal. Sem essa segunda margem, a política poderia refletir correção cadastral ou perda de cobertura.

Esse é um problema de decisão pública. Indicadores próximos da regra de gestão podem responder sem garantir o desfecho social que justifica a política. Na proteção social, \citet{{pradofirpo2026integridade}} defendem batimentos de bases e revisão orientada por risco para fortalecer integridade e eficiência do gasto. Aqui, a dimensão adicional é emprego. A meta-análise de \citet{{card2018what}} mostra que políticas ativas de mercado de trabalho tendem a ter efeitos pequenos no curto prazo e mais positivos depois de dois ou três anos, com heterogeneidade por desenho e público atendido.

Transferências condicionadas podem afetar capital humano e trabalho. Avaliações do Bolsa Família encontram ganhos de escolaridade \citep{{glewwe2012bolsa,debrauw2015schooling}} e mobilidade posterior via saída de programas sociais e acesso ao emprego formal \citep{{fassarella2024mobility}}. Para o México, \citet{{parker2023cct}} mostram efeitos de longo prazo sobre escolaridade, mobilidade e mercado de trabalho. A evidência sobre oferta de trabalho é mais cautelosa do que a narrativa de desincentivo sugere \citep{{debrauw2015labor,barbosa2014informality,banerjee2017lazy}}. Estudos recentes também encontram maior duração no emprego formal e efeitos positivos de expansão de transferências sobre emprego \citep{{santos2017duration,best2026productive}}.

Estimo um controle sintético para o município de Bento Gonçalves usando municípios da região Sul. Os desfechos principais são famílias beneficiárias e estoque de vínculos formais. O segundo é agregado e não liga famílias individuais a postos de trabalho. Ainda assim, a série de emprego ajuda a separar inclusão produtiva de revisão administrativa.

\section{{Estratégia empírica}}

A estratégia segue o controle sintético de \citet{{abadie2010synthetic}}. A ideia é comparar o município tratado com uma média ponderada de municípios não tratados que reproduza sua trajetória antes da intervenção. O efeito mensal é $\alpha_{{1t}}=Y_{{1t}}-Y^N_{{1t}}$, em que $Y^N_{{1t}}$ é aproximado por $\widehat{{Y}}^N_{{1t}}=\sum_jw_jY_{{jt}}$, com $w_j\geq0$ e $\sum_jw_j=1$. Os pesos minimizam a distância pré-tratamento usando apenas informação anterior à intervenção.

Antes de rodar o controle sintético, fixo a amostra de comparação. Para cada desfecho, entram municípios com painel mensal completo; nas séries em índice, também exijo valor positivo no mês base. O pool principal é a região Sul (RS, SC e PR), e a Tabela~\ref{{tab:results}} reporta a robustez restrita ao Rio Grande do Sul. Bento Gonçalves está dentro do suporte observado de população, PIB e PIB per capita do Sul.

A escolha de preditores segue a cautela de \citet{{ferman2020cherrypicking}}, que mostram como lags e covariáveis escolhidos após inspeção dos resultados podem abrir espaço para busca de especificações em controle sintético. Por isso, trato a etapa de seleção como desenho prévio e uso uma grade fixa de lags. Dentro do pool, ordeno até 80 municípios por preditores de pré-tratamento, comparo distância de matching e RMSPE direto, e testo 12, 15, 20 e 25 doadores. Retenho a combinação de menor RMSPE pré. Os dois desfechos usam os lags $t-1$ a $t-12$, $t-15$ e $t-18$, além de log população e log PIB. Nenhuma observação posterior a outubro de 2024 entra nessa seleção.

O tratamento operacional é novembro de 2024, mês usado pela Prefeitura como referência inicial da redução de famílias beneficiárias \citep{{prefeiturabento2025bolsafamilia}}. Os placebos reestimam cada doador retido como tratado e mantêm no gráfico os casos com MSPE pré até cinco vezes o de Bento Gonçalves. O $p_{{FP}}$ vem da rotina SCM.CS de \citet{{firpo2018synthetic}}. As bandas mostram seus conjuntos de confiança de 90\%. As figuras usam média móvel de 3 meses. Os dados combinam MDS/VISDATA, Novo CAGED e covariáveis municipais da Base dos Dados.

\begin{{table*}}[t]
\centering
\caption{{Estimativas de controle sintético para Bolsa Família e emprego formal}}
\label{{tab:results}}
\input{{tables/tab_results.tex}}
\end{{table*}}

\FloatBarrier
\vspace{{3pt}}
\section{{Resultados}}

Os painéis comparam o município de Bento Gonçalves ao seu contrafactual sintético nos dois desfechos centrais. O Bolsa Família cai {fmt0(abs(bolsa['last_effect']))} famílias no último mês ($p_{{FP}}={fmt(bolsa['p_value'], 3)}$). O estoque formal fica {fmt0(stock['last_effect'])} vínculos acima do sintético ($p_{{FP}}={fmt(stock['p_value'], 3)}$). Esse segundo resultado é compatível com maior inserção no mercado formal local no período posterior à política.

A queda no Bolsa Família equivale a {fmt_pt(bolsa_last_pct, 0)}\% do município sintético em março de 2026. O efeito no estoque formal é menor em termos proporcionais, cerca de {fmt_pt(stock_last_pct, 0)}\%, mas soma {fmt0(stock['last_effect'])} vínculos. Isso corresponde a {fmt_pt(last_ratio, 1)} vínculo formal para cada família a menos no programa no último mês. Na média pós-tratamento, a razão é próxima de {fmt_pt(mean_ratio, 1)} vínculos por família. A comparação sugere magnitudes administrativamente próximas, embora a série agregada não permita concluir que as famílias que saíram foram exatamente as contratadas.

O exercício principal retém os doadores com melhor desempenho pré-tratamento dentro do pool regional. São {fmt0(bolsa['n_donors'])} doadores e {fmt0(bolsa['n_placebos_goodfit'])} placebos para Bolsa Família, e {fmt0(stock['n_donors'])} doadores e {fmt0(stock['n_placebos_goodfit'])} placebos para estoque formal. Essa seleção melhora o ajuste visual no pré-tratamento e evita um pool amplo demais para uma aplicação municipal.

A leitura causal deve permanecer no nível agregado. A trajetória do município de Bento Gonçalves se afasta do contrafactual regional, o que enfraquece uma leitura puramente descritiva de queda mecânica no Bolsa Família. O mecanismo permanece em aberto. O aumento de estoque formal é consistente com inclusão produtiva, mas dados individuais são necessários para verificar se as famílias que saíram do programa foram absorvidas pelo mercado formal.

\section{{Considerações finais}}

Depois de novembro de 2024, o município de Bento Gonçalves registra menos famílias no Bolsa Família e mais vínculos formais do que seria esperado a partir de municípios semelhantes da região Sul. O contrafactual sintético disciplina a comparação e reduz o risco de confundir a política local com movimentos regionais comuns. A leitura mais prudente é que há evidência consistente de queda adicional no Bolsa Família e sinal compatível de melhora no mercado formal local. O desenho não identifica quem saiu do programa.

Os resultados sustentam monitoramento e investigação. Adoção em outros municípios deveria depender de replicação local, persistência dos efeitos e validação individual do mecanismo. A próxima etapa é ligar Cadastro Único, folha de pagamentos do programa, RAIS/CAGED e trajetórias de renda. Esses dados permitiriam verificar se as famílias que saíram do programa foram absorvidas pelo mercado formal, em quais ocupações, com que salários e por quanto tempo.

A validade externa depende da infraestrutura em volta da política. \citet{{fassarella2024mobility}} mostram que a mobilidade de beneficiários do Bolsa Família é territorialmente heterogênea e está associada à infraestrutura local de saúde e educação, além da atividade econômica municipal. Isso importa para interpretar o município de Bento Gonçalves. Revisão cadastral com busca ativa e encaminhamento ao trabalho deve funcionar melhor onde há serviços capazes de localizar famílias, orientar trajetórias e conectar trabalhadores a vagas formais. Em locais com mercado de trabalho frágil ou rede pública menos articulada, a mesma intervenção pode gerar saída administrativa sem inclusão produtiva.

Se o mecanismo for confirmado, a experiência oferece uma hipótese concreta de política municipal a ser adaptada em outros contextos. Até lá, políticas que afetam renda, trabalho e acesso a direitos devem ser avaliadas com contrafactuais claros e comunicadas com incerteza.

\clearpage
{main_figures}
\FloatBarrier
\clearpage
\setlength{{\bibsep}}{{0pt plus 0.1ex}}
\bibliography{{references}}

\clearpage
\onecolumn
\small
\appendix
\section*{{Apêndice A. Informações de submissão}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{2pt}}

\textbf{{Título em inglês (English title)}}

Public Policy Note: Bolsa Família exits and formal employment in the municipality of Bento Gonçalves.

\section*{{Abstract}}

In November 2024, the municipality of Bento Gonçalves launched a policy combining Bolsa Família registry review, outreach to working-age families, and referrals to formal jobs. This note estimates its aggregate effect with synthetic control. In March 2026, the municipality had {fmt0(abs(bolsa['last_effect']))} fewer families in the program than its counterfactual, and formal employment stood {fmt0(stock['last_effect'])} jobs above the synthetic series. The results indicate that beneficiary exits were accompanied by improvement in the local formal labor market, a pattern consistent with productive inclusion and relevant for scale-up and monitoring decisions.

\textbf{{Palavras-chave em inglês (Keywords)}}

Bolsa Família; synthetic control; municipal policy; formal employment.

\textbf{{ORCID}}

Victor Rangel: \href{{https://orcid.org/0000-0002-4520-2795}}{{https://orcid.org/0000-0002-4520-2795}}.

\textbf{{Declaração de contribuição dos autores}}

Este manuscrito é de autoria única. A declaração de contribuição dos autores não se aplica.

\textbf{{Conflito de interesses (Conflict of interest)}}

O autor declara não haver conflito de interesses financeiro, institucional ou pessoal relacionado a este manuscrito.

\textbf{{Declaração de disponibilidade de dados de pesquisa (Data availability statement)}}

O código, a base final usada nas estimativas principais e as instruções de reprodução estão disponíveis em \url{{https://github.com/vr-rodrigues/bento-goncalves-bolsa-familia-synth}}. Os dados brutos são provenientes de bases públicas e podem ser regenerados pelos scripts e consultas SQL do projeto.

\textbf{{Dados e uso de IA}}

Este projeto utilizou Codex como apoio operacional à organização de arquivos, automação de rotinas de coleta, execução de scripts, formatação de tabelas e checagens de reprodutibilidade. A pergunta de pesquisa, as decisões técnicas e metodológicas, as escolhas econométricas, a interpretação dos resultados e a responsabilidade por eventuais erros são integralmente do autor.

\end{{document}}
"""
    )
    (PAPER / "main.tex").write_text(main_content, encoding="utf-8")

    appendix_content = (
        preamble("Apêndice: robustez com pool RS")
        + rf"""
\begin{{document}}
\maketitle

\section{{Robustez com pool RS}}

Este apêndice reporta a robustez dos quatro desfechos principais com pool restrito ao Rio Grande do Sul. Quando a rotina de Firpo e Possebom retorna conjunto vazio para a classe linear de efeitos, o painel de IC é deixado sem banda sombreada.

\begin{{table}}[H]
\centering
\caption{{Robustez com pool RS}}
\label{{tab:appendix-results}}
\input{{tables/tab_appendix_results.tex}}
\end{{table}}

{appendix_figures}

\end{{document}}
"""
    )
    (PAPER / "appendix.tex").write_text(appendix_content, encoding="utf-8")


def write_readme() -> None:
    content = """# Nota em estilo Economics Letters

Esta pasta contem uma nota compacta em duas colunas, em portugues, com figuras montadas a partir de paineis separados. O apendice e um documento separado.

Arquivos principais:

- `main.tex`: fonte do artigo.
- `main.pdf`: relatorio compilado.
- `appendix.tex`: fonte do apendice separado.
- `appendix.pdf`: apendice compilado.
- `figures/`: paineis separados de cada figura.
- `tables/tab_results.tex`: tabela compacta de resultados.

O texto principal destaca Bolsa Familia e estoque formal de vinculos. As consultas complementares estao em `../../sql/` e usam as tabelas publicas da Base dos Dados no BigQuery.

Regenerar a partir da raiz do projeto:

```powershell
python scripts\\08_build_economics_letters_report.py
cd paper\\economics_letters_project
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode appendix.tex
```
"""
    (PAPER / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    figures_dir = PAPER / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for old_figure in figures_dir.glob("fig*.pdf"):
        old_figure.unlink()
    summary, results = run_analysis()
    order = [
        "bolsa_familia",
        "emprego_estoque",
        "emprego_admissoes",
        "emprego_desligamentos",
        "emprego_saldo",
        "taxa_admissao",
        "taxa_desligamento",
        "taxa_rotatividade",
        "empresas_abertas",
    ]
    main_outcomes = MAIN_OUTCOMES
    appendix_extra_outcomes = APPENDIX_EXTRA_OUTCOMES
    panels = ["series", "ci", "placebo"]
    for idx, outcome_key in enumerate(order, start=1):
        for panel in panels:
            if outcome_key in main_outcomes:
                main_idx = main_outcomes.index(outcome_key) + 1
                draw_panel_pdf(
                    results[f"{outcome_key}_south"],
                    PAPER
                    / "figures"
                    / f"fig_main_{main_idx:02d}_{outcome_key}_{panel}.pdf",
                    panel,
                    outcome_key,
                )
                draw_panel_pdf(
                    results[f"{outcome_key}_rs"],
                    PAPER
                    / "figures"
                    / f"fig_app-rs_{main_idx:02d}_{outcome_key}_{panel}.pdf",
                    panel,
                    outcome_key,
                )
            if outcome_key in appendix_extra_outcomes:
                app_idx = appendix_extra_outcomes.index(outcome_key) + 1
                draw_panel_pdf(
                    results[f"{outcome_key}_south"],
                    PAPER
                    / "figures"
                    / f"fig_app-extra_{app_idx:02d}_{outcome_key}_{panel}.pdf",
                    panel,
                    outcome_key,
                )
    write_tables(summary, results)
    write_rais_table()
    write_references()
    write_main_tex(summary, results)
    write_readme()
    print(PAPER)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
