from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bg_bolsa.caged import enrich_caged_with_ibge_labels, maybe_download_caged_xlsx, parse_caged_tabela8
from bg_bolsa.config import load_config


def main() -> None:
    config = load_config()
    raw_path = ROOT / "data" / "raw" / f"caged_tabelas_{config['data']['caged_latest_month']}.xlsx"
    processed_path = ROOT / "data" / "processed" / "caged_municipio_panel.csv"
    xlsx = maybe_download_caged_xlsx(config["data"]["caged_google_drive_file_id"], raw_path)
    panel = parse_caged_tabela8(xlsx)
    panel = enrich_caged_with_ibge_labels(
        panel, ROOT / "data" / "processed" / "bolsa_familia_municipio_panel.csv"
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(processed_path, index=False)
    print(f"Saved {len(panel):,} rows to {processed_path}")
    print(panel.loc[panel["id_municipio_6"] == config["treatment"]["id_municipio_mds"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
