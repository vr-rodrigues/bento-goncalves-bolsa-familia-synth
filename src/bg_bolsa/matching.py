from __future__ import annotations

import numpy as np
import pandas as pd


def units_with_complete_panel(
    panel: pd.DataFrame,
    months: list[int],
    value_col: str = "familias_bf",
    require_positive: bool = True,
) -> list[str]:
    frame = panel[panel["anomes"].isin(months)].dropna(subset=[value_col]).copy()
    if require_positive:
        frame = frame[pd.to_numeric(frame[value_col], errors="coerce") > 0]
    counts = (
        frame
        .groupby("id_municipio_6")["anomes"]
        .nunique()
    )
    return counts[counts == len(months)].index.tolist()


def build_features(
    panel: pd.DataFrame,
    pre_months: list[int],
    lagged_y_months: list[int] | None = None,
    value_col: str = "familias_bf",
    auxiliary_col: str | None = "beneficio_medio",
    require_positive: bool = True,
) -> pd.DataFrame:
    feature_rows: list[dict] = []
    lagged_y_months = lagged_y_months or []
    selected_months = [
        pre_months[0],
        pre_months[len(pre_months) // 3],
        pre_months[(2 * len(pre_months)) // 3],
        pre_months[-1],
    ]
    for municipio_id, group in panel[panel["anomes"].isin(pre_months)].groupby("id_municipio_6"):
        group = group.sort_values("anomes")
        if group["anomes"].nunique() != len(pre_months):
            continue
        y = group[value_col].astype(float).to_numpy()
        if np.any(~np.isfinite(y)):
            continue
        if require_positive and np.any(y <= 0):
            continue
        aux = None
        if auxiliary_col and auxiliary_col in group.columns:
            aux = group[auxiliary_col].astype(float).to_numpy()
        x = np.arange(len(y), dtype=float)
        slope_y = np.log1p(y) if require_positive else y
        slope = float(np.polyfit(x, slope_y, 1)[0])
        first = float(y[0])
        last = float(y[-1])
        month_values = group.set_index("anomes")[value_col].astype(float)
        pct_change = (last / first) - 1.0 if first != 0 else np.nan
        record = {
            "id_municipio_6": municipio_id,
            "pre_mean_y": float(np.mean(y)),
            "pre_log_mean_y": float(np.mean(np.log1p(y))) if require_positive else float(np.mean(y)),
            "pre_log_slope": slope,
            "pre_pct_change": float(pct_change),
            "pre_sd_log": float(np.std(np.log1p(y))) if require_positive else float(np.std(y)),
            "pre_last_y": last,
        }
        if not require_positive:
            record["pre_level_change"] = last - first
            record["pre_level_min"] = float(np.min(y))
            record["pre_level_max"] = float(np.max(y))
        if aux is not None:
            record[f"pre_mean_{auxiliary_col}"] = float(np.nanmean(aux))
        for month in selected_months:
            value = month_values.loc[month]
            record[f"y_{month}"] = float(value)
        for month in lagged_y_months:
            value = month_values.loc[month]
            if require_positive:
                record[f"lag_y_index_{month}"] = float(100.0 * value / last)
            else:
                record[f"lag_y_level_{month}"] = float(value)
        meta_cols = ["municipio", "sigla_uf", "uf", "regiao"]
        for col in meta_cols:
            if col in group.columns:
                record[col] = group[col].iloc[0]
        feature_rows.append(record)
    return pd.DataFrame(feature_rows)


def match_nearest(
    features: pd.DataFrame,
    treated_id: str,
    candidate_ids: list[str],
    k: int,
) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in features.columns
        if c not in {"id_municipio_6", "municipio", "sigla_uf", "uf", "regiao"}
        and pd.api.types.is_numeric_dtype(features[c])
    ]
    pool = features[features["id_municipio_6"].isin(candidate_ids + [treated_id])].copy()
    treated = pool[pool["id_municipio_6"] == treated_id]
    if treated.empty:
        raise ValueError(f"Treated municipality {treated_id} not found in features")
    candidates = pool[pool["id_municipio_6"].isin(candidate_ids)].copy()
    means = candidates[numeric_cols].mean()
    stds = candidates[numeric_cols].std().replace(0, 1)
    normalized_treated = ((treated[numeric_cols].iloc[0] - means) / stds).fillna(0)
    normalized_candidates = ((candidates[numeric_cols] - means) / stds).fillna(0)
    treated_vec = normalized_treated.to_numpy(dtype=float)
    cand_mat = normalized_candidates.to_numpy(dtype=float)
    distances = np.sqrt(np.nanmean((cand_mat - treated_vec) ** 2, axis=1))
    candidates["match_distance"] = distances
    return candidates.sort_values("match_distance").head(k).reset_index(drop=True)
