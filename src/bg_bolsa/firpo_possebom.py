from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bg_bolsa.synth import run_synth


@dataclass(frozen=True)
class SCMConfidenceSet:
    lower: np.ndarray
    upper: np.ndarray
    lb_param: float
    ub_param: float
    initial_param: float
    significance: float
    effect_type: str
    zero_p_value: float
    zero_reject: bool


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _null_effect(length: int, t0: int, param: float, effect_type: str) -> np.ndarray:
    effect = np.zeros(length, dtype=float)
    post_len = length - t0
    if effect_type == "constant":
        effect[t0:] = param
    elif effect_type == "linear":
        effect[t0:] = param * np.arange(1, post_len + 1, dtype=float)
    else:
        raise ValueError("effect_type must be 'linear' or 'constant'.")
    return effect


def _rmspe_ratio(gaps: np.ndarray, t0: int) -> float:
    pre = float(np.mean(np.square(gaps[:t0])))
    post = float(np.mean(np.square(gaps[t0:])))
    if pre <= 1e-24:
        return np.inf
    return post / pre


def _test_null(
    y_mat: np.ndarray,
    weights_mat: np.ndarray,
    treated: int,
    t0: int,
    null_effect: np.ndarray,
    phi: float,
    v: np.ndarray,
    significance: float,
) -> tuple[bool, float, np.ndarray]:
    n_units = y_mat.shape[1]
    stats = np.empty(n_units, dtype=float)
    unit_positions = np.arange(n_units)
    for j in range(n_units):
        controls = unit_positions[unit_positions != j]
        if j == treated:
            y1 = y_mat[:, treated]
            y0 = y_mat[:, controls]
        else:
            y1 = y_mat[:, j] + null_effect
            y0 = y_mat[:, controls].copy()
            treated_control_pos = np.where(controls == treated)[0]
            if treated_control_pos.size:
                y0[:, treated_control_pos[0]] = y0[:, treated_control_pos[0]] - null_effect
        gaps = y1 - y0 @ weights_mat[:, j] - null_effect
        stats[j] = _rmspe_ratio(gaps, t0)

    ranks = _average_ranks(stats)
    probs = np.exp(phi * v)
    probs = probs / probs.sum()
    p_value = float(probs @ (ranks >= ranks[treated]).astype(float))
    return p_value <= significance, p_value, stats


def scm_confidence_set(
    y_mat: np.ndarray,
    weights_mat: np.ndarray,
    treated: int,
    t0: int,
    *,
    phi: float = 0.0,
    v: np.ndarray | None = None,
    precision: int = 30,
    effect_type: str = "linear",
    significance: float | None = None,
    alpha: float = 0.10,
) -> SCMConfidenceSet:
    """Python port of Firpo-Possebom's SCM.CS routine.

    Arguments follow the R supplement: rows of ``y_mat`` are time periods,
    columns are units, and each column of ``weights_mat`` stores weights for
    the corresponding unit's synthetic control, ordered by all other units.
    ``t0`` is the number of pre-treatment periods.
    """
    y_mat = np.asarray(y_mat, dtype=float)
    weights_mat = np.asarray(weights_mat, dtype=float)
    if y_mat.ndim != 2:
        raise ValueError("y_mat must be a 2D matrix.")
    n_periods, n_units = y_mat.shape
    if weights_mat.shape != (n_units - 1, n_units):
        raise ValueError("weights_mat must have shape (n_units - 1, n_units).")
    if treated < 0 or treated >= n_units:
        raise ValueError("treated index is out of range.")
    if t0 <= 0 or t0 >= n_periods:
        raise ValueError("t0 must be between 1 and n_periods - 1.")
    if effect_type not in {"linear", "constant"}:
        raise ValueError("effect_type must be 'linear' or 'constant'.")
    if phi < 0:
        raise ValueError("phi must be non-negative.")
    if precision < 0 or int(precision) != precision:
        raise ValueError("precision must be a non-negative integer.")
    if v is None:
        v = np.zeros(n_units, dtype=float)
    else:
        v = np.asarray(v, dtype=float).reshape(-1)
    if v.shape[0] != n_units:
        raise ValueError("v must have one entry per unit.")
    if np.any((v < 0) | (v > 1)):
        raise ValueError("v entries must lie in [0, 1].")
    if significance is None:
        significance = alpha
    if not 0 < significance < 1:
        raise ValueError("significance must lie in (0, 1).")

    controls = [i for i in range(n_units) if i != treated]
    gaps = y_mat[:, treated] - y_mat[:, controls] @ weights_mat[:, treated]
    post_len = n_periods - t0
    if effect_type == "constant":
        param = float(np.mean(gaps[t0:]))
    else:
        param = float(gaps[-1] / post_len)
    if abs(param) < 1e-12:
        param = 1e-12
    sign = float(np.sign(param))
    ub = param
    lb = param
    attempt_u = True
    attempt_l = True

    for power in range(int(precision) + 1):
        step = 0.5**power
        reject_u = False
        reject_l = False
        while not reject_u:
            null = _null_effect(n_periods, t0, ub, effect_type)
            reject_u, _, _ = _test_null(
                y_mat, weights_mat, treated, t0, null, phi, v, significance
            )
            if not reject_u:
                ub = param * (ub / param + sign * step)
                attempt_u = False
            else:
                if attempt_u:
                    raise RuntimeError("Empty confidence set for the requested class of effects.")
                ub = param * (ub / param - sign * step)
            if abs(ub) > 100 * abs(param):
                raise RuntimeError("Upper bound was not found.")

        while not reject_l:
            null = _null_effect(n_periods, t0, lb, effect_type)
            reject_l, _, _ = _test_null(
                y_mat, weights_mat, treated, t0, null, phi, v, significance
            )
            if not reject_l:
                lb = param * (lb / param - sign * step)
                attempt_l = False
            else:
                if attempt_l:
                    raise RuntimeError("Empty confidence set for the requested class of effects.")
                lb = param * (lb / param + sign * step)
            if abs(lb) > 100 * abs(param):
                raise RuntimeError("Lower bound was not found.")

    lower = _null_effect(n_periods, t0, lb, effect_type)
    upper = _null_effect(n_periods, t0, ub, effect_type)
    low = np.minimum(lower, upper)
    high = np.maximum(lower, upper)
    zero_reject, zero_p_value, _ = _test_null(
        y_mat,
        weights_mat,
        treated,
        t0,
        np.zeros(n_periods, dtype=float),
        phi,
        v,
        significance,
    )
    return SCMConfidenceSet(
        lower=low,
        upper=high,
        lb_param=lb,
        ub_param=ub,
        initial_param=param,
        significance=float(significance),
        effect_type=effect_type,
        zero_p_value=zero_p_value,
        zero_reject=zero_reject,
    )


def _smooth_matrix(y_mat: np.ndarray, window: int | None) -> np.ndarray:
    if not window or window <= 1:
        return y_mat
    return np.column_stack(
        [
            pd.Series(y_mat[:, j], dtype=float).rolling(window, min_periods=1).mean().to_numpy()
            for j in range(y_mat.shape[1])
        ]
    )


def build_scm_matrices(
    panel: pd.DataFrame,
    unit_ids: list[str],
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    value_col: str,
    scale_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    pivot = panel.pivot_table(
        index="anomes", columns="id_municipio_6", values=value_col, aggfunc="first"
    ).sort_index()
    counts = pivot.loc[all_months, unit_ids].astype(float)
    if scale_mode == "index":
        baseline = counts.loc[baseline_month]
        if np.any(~np.isfinite(baseline.to_numpy(dtype=float))) or np.any(
            baseline.to_numpy(dtype=float) == 0
        ):
            raise ValueError("Index scale requires finite, non-zero baseline values.")
        y_mat = counts.divide(baseline, axis=1).to_numpy(dtype=float) * 100.0
    elif scale_mode == "level":
        y_mat = counts.to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    n_units = len(unit_ids)
    weights_mat = np.empty((n_units - 1, n_units), dtype=float)
    for j, unit_id in enumerate(unit_ids):
        controls = [u for u in unit_ids if u != unit_id]
        result = run_synth(
            panel,
            unit_id,
            controls,
            pre_months,
            all_months,
            baseline_month,
            value_col=value_col,
            scale_mode=scale_mode,
        )
        weights = result["weights"].set_index("id_municipio_6").reindex(controls)["weight"]
        weights_mat[:, j] = weights.to_numpy(dtype=float)
    return y_mat, weights_mat


def confidence_set_for_panel(
    panel: pd.DataFrame,
    treated_id: str,
    donor_ids: list[str],
    pre_months: list[int],
    all_months: list[int],
    baseline_month: int,
    value_col: str,
    scale_mode: str,
    *,
    alpha: float = 0.10,
    precision: int = 30,
    effect_type: str = "linear",
    ma_window: int | None = 3,
) -> tuple[pd.DataFrame, SCMConfidenceSet]:
    unit_ids = [treated_id] + donor_ids
    y_mat, weights_mat = build_scm_matrices(
        panel, unit_ids, pre_months, all_months, baseline_month, value_col, scale_mode
    )
    y_for_ci = _smooth_matrix(y_mat, ma_window)
    cs = scm_confidence_set(
        y_for_ci,
        weights_mat,
        treated=0,
        t0=len(pre_months),
        alpha=alpha,
        precision=precision,
        effect_type=effect_type,
    )
    if scale_mode == "index":
        pivot = panel.pivot_table(
            index="anomes", columns="id_municipio_6", values=value_col, aggfunc="first"
        ).sort_index()
        conversion = float(pivot.loc[baseline_month, treated_id]) / 100.0
    else:
        conversion = 1.0
    bounds = pd.DataFrame(
        {
            "anomes": all_months,
            "lower_index": cs.lower,
            "upper_index": cs.upper,
            "lower_value": cs.lower * conversion,
            "upper_value": cs.upper * conversion,
            "fp_initial_param": cs.initial_param,
            "fp_lb_param": cs.lb_param,
            "fp_ub_param": cs.ub_param,
            "fp_significance": cs.significance,
            "fp_effect_type": cs.effect_type,
        }
    )
    return bounds, cs
