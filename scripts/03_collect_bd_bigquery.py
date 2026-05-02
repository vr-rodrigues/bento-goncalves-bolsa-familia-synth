from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "upa-research")

JOBS = [
    (
        ROOT / "sql" / "municipal_covariates.sql",
        ROOT / "data" / "processed" / "municipal_covariates.csv",
    ),
    (
        ROOT / "sql" / "rais_income_municipio.sql",
        ROOT / "data" / "processed" / "rais_income_municipio_panel.csv",
    ),
    (
        ROOT / "sql" / "cnpj_births_municipio_monthly.sql",
        ROOT / "data" / "processed" / "cnpj_births_municipio_monthly.csv",
    ),
]


def has_gcloud_auth() -> bool:
    gcloud = shutil.which("gcloud")
    if not gcloud:
        return False
    proc = subprocess.run(
        [gcloud, "auth", "list", "--format=json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    try:
        accounts = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return False
    return any(account.get("status") == "ACTIVE" for account in accounts)


def run_bq_query(sql_path: Path, output_path: Path) -> None:
    bq = shutil.which("bq")
    if not bq:
        raise RuntimeError("bq CLI not found. Install Google Cloud SDK first.")
    sql = sql_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            bq,
            "query",
            f"--project_id={PROJECT_ID}",
            "--use_legacy_sql=false",
            "--format=csv",
            "--max_rows=1000000",
        ],
        input=sql,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    output_path.write_text(proc.stdout, encoding="utf-8")
    print(f"Saved {output_path}")


def main() -> None:
    if not has_gcloud_auth():
        print(
            "\n".join(
                [
                    "Google Cloud credentials are not active in this machine.",
                    f"Project configured for billing: {PROJECT_ID}",
                    "Run:",
                    "",
                    "  gcloud auth login",
                    f"  gcloud config set project {PROJECT_ID}",
                    "",
                    "Then rerun:",
                    "",
                    f"  \"{sys.executable}\" scripts/03_collect_bd_bigquery.py",
                ]
            )
        )
        return
    for sql_path, output_path in JOBS:
        run_bq_query(sql_path, output_path)


if __name__ == "__main__":
    main()
