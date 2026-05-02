from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd


MONTHS_PT = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "março": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

VARIABLES = {
    "estoque": "estoque",
    "admissoes": "admissoes",
    "admissões": "admissoes",
    "desligamentos": "desligamentos",
    "saldos": "saldo",
    "saldo": "saldo",
    "variacao relativa (%)": "variacao_relativa",
    "variação relativa (%)": "variacao_relativa",
}


def _strip_accents(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char)
    )


def _clean_text(value: object) -> str:
    return _strip_accents(str(value).replace("\n", " ").strip().lower())


def parse_month_label(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("\n", " ").strip()
    if "/" not in text:
        return None
    month_name, year_text = text.split("/", 1)
    month = MONTHS_PT.get(month_name.strip().lower())
    try:
        year = int(year_text[:4])
    except ValueError:
        return None
    if month is None:
        return None
    return year * 100 + month


def maybe_download_caged_xlsx(file_id: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 1_000_000:
        return output_path

    deps = output_path.parents[2] / ".deps"
    if deps.exists():
        sys.path.insert(0, str(deps.resolve()))
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required to download the Novo CAGED workbook. "
            "Install project dependencies or place the workbook at data/raw/caged_tabelas_202603.xlsx."
        ) from exc
    gdown.download(id=file_id, output=str(output_path), quiet=False, use_cookies=False)
    return output_path


def parse_caged_tabela8(xlsx_path: Path) -> pd.DataFrame:
    wide = pd.read_excel(xlsx_path, sheet_name="Tabela 8", header=None)
    month_by_col: dict[int, int] = {}
    current_month: int | None = None
    for col in range(4, wide.shape[1]):
        header_value = wide.iat[4, col]
        parsed = parse_month_label(header_value)
        if parsed is not None:
            current_month = parsed
        elif header_value is not None and not pd.isna(header_value):
            current_month = None
        variable = VARIABLES.get(_clean_text(wide.iat[5, col]))
        if current_month is not None and variable is not None:
            month_by_col[col] = current_month

    records: list[dict] = []
    data = wide.iloc[6:].copy()
    data = data[pd.to_numeric(data.iloc[:, 2], errors="coerce").notna()]
    for _, row in data.iterrows():
        municipio_id = str(int(row.iloc[2])).zfill(6)
        base = {
            "sigla_uf": row.iloc[1],
            "id_municipio_6": municipio_id,
            "municipio_caged": row.iloc[3],
        }
        by_month: dict[int, dict] = {}
        for col, anomes in month_by_col.items():
            variable = VARIABLES.get(_clean_text(wide.iat[5, col]))
            if variable is None:
                continue
            by_month.setdefault(anomes, dict(base, anomes=anomes))[variable] = row.iloc[col]
        records.extend(by_month.values())

    out = pd.DataFrame.from_records(records)
    for col in ["estoque", "admissoes", "desligamentos", "saldo", "variacao_relativa"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["id_municipio_6", "anomes"]).reset_index(drop=True)


def enrich_caged_with_ibge_labels(caged: pd.DataFrame, bolsa_panel_path: Path) -> pd.DataFrame:
    if not bolsa_panel_path.exists():
        return caged
    labels = pd.read_csv(bolsa_panel_path, dtype={"id_municipio_6": str})[
        ["id_municipio_6", "municipio", "uf", "regiao", "id_microrregiao", "microrregiao"]
    ].drop_duplicates()
    labels["id_municipio_6"] = labels["id_municipio_6"].astype(str).str.zfill(6)
    out = caged.merge(labels, on="id_municipio_6", how="left")
    out["municipio"] = out["municipio"].fillna(out["municipio_caged"])
    return out
