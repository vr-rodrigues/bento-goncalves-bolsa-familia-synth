from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bg_bolsa.config import load_config
from bg_bolsa.data_sources import collect_mds_bolsa, fetch_ibge_municipalities, prepare_bolsa_panel


def main() -> None:
    config = load_config()
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    bolsa_raw = collect_mds_bolsa(raw_dir, config["data"]["mds_years"])
    municipios = fetch_ibge_municipalities(raw_dir)
    panel = prepare_bolsa_panel(bolsa_raw, municipios)

    panel_path = processed_dir / "bolsa_familia_municipio_panel.csv"
    panel.to_csv(panel_path, index=False)
    print(f"Saved {len(panel):,} rows to {panel_path}")
    print(
        panel.loc[panel["id_municipio_6"] == config["treatment"]["id_municipio_mds"]]
        .tail(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
