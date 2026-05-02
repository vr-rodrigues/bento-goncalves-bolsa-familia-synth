from __future__ import annotations

import io
import gzip
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


MDS_ENDPOINT = "https://aplicacoes.mds.gov.br/sagi/servicos/misocial"
IBGE_MUNICIPIOS_ENDPOINT = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "bg-bolsa-synth/0.1"})
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = response.read()
    if payload[:2] == b"\x1f\x8b":
        return gzip.decompress(payload)
    return payload


def fetch_mds_bolsa_year(year: int) -> pd.DataFrame:
    params = [
        ("q", "*"),
        ("fq", f"anomes_s:{year}*"),
        ("fq", "tipo_s:mes_mu"),
        ("wt", "csv"),
        (
            "fl",
            "ibge:codigo_ibge,anomes:anomes_s,"
            "qtd_familias_beneficiarias_bolsa_familia,"
            "valor_repassado_bolsa_familia",
        ),
        ("rows", "10000000"),
        ("sort", "anomes_s asc, codigo_ibge asc"),
    ]
    url = f"{MDS_ENDPOINT}?{urllib.parse.urlencode(params)}"
    payload = _download(url)
    df = pd.read_csv(io.BytesIO(payload))
    if df.empty:
        raise RuntimeError(f"MDS returned no rows for {year}")
    df["ibge"] = df["ibge"].astype(str).str.zfill(6)
    df["anomes"] = df["anomes"].astype(int)
    df["qtd_familias_beneficiarias_bolsa_familia"] = pd.to_numeric(
        df["qtd_familias_beneficiarias_bolsa_familia"], errors="coerce"
    )
    df["valor_repassado_bolsa_familia"] = pd.to_numeric(
        df["valor_repassado_bolsa_familia"], errors="coerce"
    )
    return df


def collect_mds_bolsa(raw_dir: Path, years: list[int], refresh: bool = False) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for year in years:
        path = raw_dir / f"mds_bolsa_familia_municipio_{year}.csv"
        if refresh or not path.exists():
            df_year = fetch_mds_bolsa_year(year)
            df_year.to_csv(path, index=False)
        else:
            df_year = pd.read_csv(path, dtype={"ibge": str})
            df_year["ibge"] = df_year["ibge"].astype(str).str.zfill(6)
        frames.append(df_year)
    return pd.concat(frames, ignore_index=True)


def fetch_ibge_municipalities(raw_dir: Path, refresh: bool = False) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "ibge_municipios.json"
    if refresh or not path.exists():
        payload = _download(IBGE_MUNICIPIOS_ENDPOINT)
        path.write_bytes(payload)
    payload = path.read_bytes()
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
        path.write_bytes(payload)
    rows = json.loads(payload.decode("utf-8"))
    records: list[dict] = []
    for row in rows:
        micro = row.get("microrregiao") or {}
        meso = micro.get("mesorregiao") or {}
        uf = meso.get("UF") or {}
        regiao = uf.get("regiao") or {}
        id7 = str(row["id"]).zfill(7)
        records.append(
            {
                "id_municipio_7": id7,
                "id_municipio_6": id7[:6],
                "municipio": row.get("nome"),
                "sigla_uf": uf.get("sigla"),
                "uf": uf.get("nome"),
                "regiao": regiao.get("nome"),
                "id_microrregiao": micro.get("id"),
                "microrregiao": micro.get("nome"),
                "id_mesorregiao": meso.get("id"),
                "mesorregiao": meso.get("nome"),
            }
        )
    return pd.DataFrame.from_records(records)


def prepare_bolsa_panel(df: pd.DataFrame, municipios: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    out["id_municipio_6"] = out["ibge"].astype(str).str.zfill(6)
    out["familias_bf"] = pd.to_numeric(
        out["qtd_familias_beneficiarias_bolsa_familia"], errors="coerce"
    )
    out["valor_bf"] = pd.to_numeric(out["valor_repassado_bolsa_familia"], errors="coerce")
    out["beneficio_medio"] = out["valor_bf"] / out["familias_bf"].where(out["familias_bf"] != 0)
    out = out[["id_municipio_6", "anomes", "familias_bf", "valor_bf", "beneficio_medio"]]
    if municipios is not None:
        out = out.merge(municipios, on="id_municipio_6", how="left")
    return out.sort_values(["id_municipio_6", "anomes"]).reset_index(drop=True)
