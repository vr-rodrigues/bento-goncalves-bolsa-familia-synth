from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "economics_letters"
FINAL = ROOT / "data" / "final"

OUTCOMES = {
    "bolsa_familia": {
        "label": "Familias beneficiarias do Bolsa Familia",
        "unit": "familias",
    },
    "emprego_estoque": {
        "label": "Estoque de vinculos formais",
        "unit": "vinculos",
    },
}
POOLS = {
    "south": "Regiao Sul (RS, SC e PR)",
    "rs": "Rio Grande do Sul",
}


def read_timeseries() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for outcome, meta in OUTCOMES.items():
        for pool, pool_label in POOLS.items():
            path = OUT / f"{outcome}_{pool}_timeseries.csv"
            frame = pd.read_csv(path)
            frame.insert(0, "pool_label", pool_label)
            frame.insert(0, "pool", pool)
            frame.insert(0, "outcome_unit", meta["unit"])
            frame.insert(0, "outcome_label", meta["label"])
            frame.insert(0, "outcome", outcome)
            frames.append(frame)
    cols = [
        "outcome",
        "outcome_label",
        "outcome_unit",
        "pool",
        "pool_label",
        "anomes",
        "actual_value",
        "synth_value",
        "effect_value",
        "actual_index",
        "synth_index",
        "effect_index",
    ]
    return pd.concat(frames, ignore_index=True)[cols]


def read_estimates() -> pd.DataFrame:
    summary = pd.read_csv(OUT / "summary.csv")
    keep = summary[
        summary["outcome"].isin(OUTCOMES.keys()) & summary["pool"].isin(POOLS.keys())
    ].copy()
    keep["outcome_label"] = keep["outcome"].map(lambda x: OUTCOMES[x]["label"])
    keep["outcome_unit"] = keep["outcome"].map(lambda x: OUTCOMES[x]["unit"])
    keep["pool_label"] = keep["pool"].map(POOLS)
    cols = [
        "outcome",
        "outcome_label",
        "outcome_unit",
        "pool",
        "pool_label",
        "matching_spec",
        "matching_lags",
        "n_donors",
        "donor_selection_order",
        "baseline_covariates",
        "scale_mode",
        "pre_rmspe",
        "post_pre_ratio",
        "p_value",
        "mean_post_effect",
        "last_actual",
        "last_synth",
        "last_effect",
        "n_placebos_goodfit",
        "loo_min",
        "loo_max",
        "fp_ci_lb_param",
        "fp_ci_ub_param",
        "fp_ci_initial_param",
        "fp_ci_significance",
    ]
    return keep[cols].sort_values(["outcome", "pool"]).reset_index(drop=True)


def read_weights() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for outcome, meta in OUTCOMES.items():
        for pool, pool_label in POOLS.items():
            path = OUT / f"{outcome}_{pool}_weights.csv"
            frame = pd.read_csv(path)
            frame.insert(0, "pool_label", pool_label)
            frame.insert(0, "pool", pool)
            frame.insert(0, "outcome_unit", meta["unit"])
            frame.insert(0, "outcome_label", meta["label"])
            frame.insert(0, "outcome", outcome)
            frames.append(frame)
    cols = [
        "outcome",
        "outcome_label",
        "outcome_unit",
        "pool",
        "pool_label",
        "id_municipio_6",
        "municipio",
        "sigla_uf",
        "weight",
    ]
    weights = pd.concat(frames, ignore_index=True)[cols]
    return weights.sort_values(["outcome", "pool", "weight"], ascending=[True, True, False])


def write_readme() -> None:
    content = """# Base final usada na nota

Esta pasta contem a versao final leve dos dados usados para auditar a nota principal.

Arquivos:

- `final_timeseries.csv`: series observadas, sinteticas e gaps para Bolsa Familia e estoque formal, nos pools Sul e RS.
- `final_estimates.csv`: estimativas da Tabela 1 e metadados de especificacao.
- `final_weights.csv`: pesos dos municipios doadores em cada controle sintetico.

Observacoes:

- As series sao medias moveis de 3 meses, como no artigo.
- O tratamento operacional e novembro de 2024.
- Os dados brutos e paineis intermediarios ficam fora do Git por tamanho e por serem regeneraveis pelos scripts do projeto.
"""
    (FINAL / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    read_timeseries().to_csv(FINAL / "final_timeseries.csv", index=False)
    read_estimates().to_csv(FINAL / "final_estimates.csv", index=False)
    read_weights().to_csv(FINAL / "final_weights.csv", index=False)
    write_readme()
    print(FINAL)


if __name__ == "__main__":
    main()
