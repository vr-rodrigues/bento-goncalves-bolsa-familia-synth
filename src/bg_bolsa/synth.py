from __future__ import annotations

import numpy as np
import pandas as pd


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    if v.size == 1:
        return np.array([1.0])
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, v.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(v.size, 1.0 / v.size)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0)


def fit_weights(x_pre: np.ndarray, y_pre: np.ndarray, max_iter: int = 350) -> np.ndarray:
    n_donors = x_pre.shape[1]
    w = np.full(n_donors, 1.0 / n_donors)
    gram = (x_pre.T @ x_pre) / max(1, x_pre.shape[0])
    eigmax = float(np.linalg.eigvalsh(gram).max()) if n_donors else 1.0
    step = 1.0 / (2.0 * eigmax + 1e-9)
    last_loss = np.inf
    for _ in range(max_iter):
        residual = x_pre @ w - y_pre
        grad = (2.0 / max(1, x_pre.shape[0])) * (x_pre.T @ residual)
        w = project_to_simplex(w - step * grad)
        loss = float(np.mean((x_pre @ w - y_pre) ** 2))
        if abs(last_loss - loss) < 1e-12:
            break
        last_loss = loss
    return w


def _rmspe(series: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(series.to_numpy(dtype=float)))))


def run_synth(
    panel: pd.DataFrame,
    treated_id: str,
    donor_ids: list[str],
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    value_col: str = "familias_bf",
    value_label: str = "families",
    scale_mode: str = "index",
) -> dict:
    pivot = panel.pivot_table(
        index="anomes", columns="id_municipio_6", values=value_col, aggfunc="first"
    ).sort_index()
    required_units = [treated_id] + donor_ids
    counts = pivot.loc[all_months, required_units].copy()
    baseline = counts.loc[baseline_month]
    if scale_mode == "index":
        if np.any(~np.isfinite(baseline.to_numpy(dtype=float))) or np.any(
            baseline.to_numpy(dtype=float) == 0
        ):
            raise ValueError("Index scale requires finite, non-zero baseline values.")
        scaled = counts.divide(baseline, axis=1) * 100.0
    elif scale_mode == "level":
        scaled = counts.copy()
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    x_pre = scaled.loc[pre_months, donor_ids].to_numpy(dtype=float)
    y_pre = scaled.loc[pre_months, treated_id].to_numpy(dtype=float)
    weights = fit_weights(x_pre, y_pre)
    synth_scaled = scaled.loc[all_months, donor_ids].to_numpy(dtype=float) @ weights
    actual_scaled = scaled.loc[all_months, treated_id].to_numpy(dtype=float)
    actual_count = counts.loc[all_months, treated_id].to_numpy(dtype=float)
    if scale_mode == "index":
        treated_baseline = float(baseline.loc[treated_id])
        synth_count = synth_scaled * treated_baseline / 100.0
    else:
        synth_count = synth_scaled
    out = pd.DataFrame(
        {
            "anomes": all_months,
            "actual_index": actual_scaled,
            "synth_index": synth_scaled,
            "effect_index": actual_scaled - synth_scaled,
            "actual_value": actual_count,
            "synth_value": synth_count,
            "effect_value": actual_count - synth_count,
        }
    )
    out[f"actual_{value_label}"] = out["actual_value"]
    out[f"synth_{value_label}"] = out["synth_value"]
    out[f"effect_{value_label}"] = out["effect_value"]
    pre_mask = out["anomes"].isin(pre_months)
    post_mask = ~pre_mask
    metrics = {
        "pre_rmspe_index": _rmspe(out.loc[pre_mask, "effect_index"]),
        "post_rmspe_index": _rmspe(out.loc[post_mask, "effect_index"]),
        "post_pre_rmspe_ratio": _rmspe(out.loc[post_mask, "effect_index"])
        / max(_rmspe(out.loc[pre_mask, "effect_index"]), 1e-12),
        "mean_post_effect_index": float(out.loc[post_mask, "effect_index"].mean()),
        "mean_post_effect_value": float(out.loc[post_mask, "effect_value"].mean()),
        "last_effect_index": float(out.iloc[-1]["effect_index"]),
        "last_effect_value": float(out.iloc[-1]["effect_value"]),
        "last_actual_value": float(out.iloc[-1]["actual_value"]),
        "last_synth_value": float(out.iloc[-1]["synth_value"]),
        "cumulative_effect_value_month": float(out.loc[post_mask, "effect_value"].sum()),
    }
    metrics[f"mean_post_effect_{value_label}"] = metrics["mean_post_effect_value"]
    metrics[f"last_effect_{value_label}"] = metrics["last_effect_value"]
    metrics[f"last_actual_{value_label}"] = metrics["last_actual_value"]
    metrics[f"last_synth_{value_label}"] = metrics["last_synth_value"]
    metrics[f"cumulative_effect_{value_label}_month"] = metrics["cumulative_effect_value_month"]
    weights_df = pd.DataFrame({"id_municipio_6": donor_ids, "weight": weights})
    return {
        "timeseries": out,
        "weights": weights_df,
        "metrics": metrics,
        "scale_mode": scale_mode,
    }


def placebo_ratios(
    panel: pd.DataFrame,
    donor_ids: list[str],
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    value_col: str = "familias_bf",
    scale_mode: str = "index",
) -> pd.DataFrame:
    rows: list[dict] = []
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
        except (ValueError, np.linalg.LinAlgError, KeyError):
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "id_municipio_6": placebo_id,
                "post_pre_rmspe_ratio": metrics["post_pre_rmspe_ratio"],
                "mean_post_effect_index": metrics["mean_post_effect_index"],
                "pre_rmspe_index": metrics["pre_rmspe_index"],
            }
        )
    return pd.DataFrame(rows)


def leave_one_out(
    panel: pd.DataFrame,
    treated_id: str,
    donor_ids: list[str],
    weights: pd.DataFrame,
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    value_col: str = "familias_bf",
    min_weight: float = 0.02,
    scale_mode: str = "index",
) -> pd.DataFrame:
    rows: list[dict] = []
    influential = weights.loc[weights["weight"] >= min_weight, "id_municipio_6"].tolist()
    for omitted in influential:
        controls = [d for d in donor_ids if d != omitted]
        if len(controls) < 2:
            continue
        result = run_synth(
            panel,
            treated_id,
            controls,
            pre_months,
            all_months,
            baseline_month,
            value_col=value_col,
            scale_mode=scale_mode,
        )
        metrics = result["metrics"]
        rows.append(
            {
                "omitted_id_municipio_6": omitted,
                "omitted_weight": float(
                    weights.loc[weights["id_municipio_6"] == omitted, "weight"].iloc[0]
                ),
                "pre_rmspe_index": metrics["pre_rmspe_index"],
                "post_pre_rmspe_ratio": metrics["post_pre_rmspe_ratio"],
                "mean_post_effect_index": metrics["mean_post_effect_index"],
                "last_effect_index": metrics["last_effect_index"],
                "last_effect_value": metrics["last_effect_value"],
            }
        )
    return pd.DataFrame(rows)


def in_time_placebo(
    panel: pd.DataFrame,
    treated_id: str,
    donor_ids: list[str],
    fake_pre_months: list[int],
    fake_post_months: list[int],
    baseline_month: int,
    value_col: str = "familias_bf",
    scale_mode: str = "index",
) -> dict:
    all_months = fake_pre_months + fake_post_months
    return run_synth(
        panel,
        treated_id,
        donor_ids,
        fake_pre_months,
        all_months,
        baseline_month,
        value_col=value_col,
        scale_mode=scale_mode,
    )
