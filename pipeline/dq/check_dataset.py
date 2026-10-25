"""Data Quality проверки датасета обучения и публикация HTML-отчёта."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from airflow.exceptions import AirflowSkipException

from pipeline.settings import CURRENT_DATASET, DQ_REPORTS_DIR

logger = logging.getLogger(__name__)


def _check(checks: list, name: str, passed: bool, details: str) -> None:
    checks.append({"rule": name, "status": "PASS" if passed else "FAIL", "details": details})


def _render_html(checks: list, dataset_path: Path, generated_at: str) -> str:
    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    rows = []
    for c in checks:
        cls = "pass" if c["status"] == "PASS" else "fail"
        rows.append(
            f"<tr><td>{c['rule']}</td>"
            f"<td class='{cls}'>{c['status']}</td>"
            f"<td>{c['details']}</td></tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Data Quality Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
th {{ background: #f2f2f2; }}
.pass {{ color: #1e7e34; font-weight: bold; }}
.fail {{ color: #dc3545; font-weight: bold; }}
.meta {{ margin-bottom: 16px; color: #555; }}
.overall {{ font-size: 18px; font-weight: bold; }}
</style>
</head>
<body>
<h1>Data Quality Report</h1>
<div class="meta">
  <div>Dataset: {dataset_path}</div>
  <div>Generated at (UTC): {generated_at}</div>
  <div class="overall">Overall: {overall}</div>
</div>
<table>
  <thead><tr><th>Rule</th><th>Status</th><th>Details</th></tr></thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
</body>
</html>"""


def run_data_quality() -> dict:
    """Проверяет текущий датасет обучения и публикует HTML-отчёт."""
    if not CURRENT_DATASET.exists():
        raise AirflowSkipException(f"Датасет не найден: {CURRENT_DATASET}")

    df = pd.read_parquet(CURRENT_DATASET)
    checks: list = []

    # 1. Датасет не пустой
    _check(checks, "Dataset is not empty", len(df) > 0, f"rows={len(df)}")

    critical = ["transaction_time", "target", "amount", "merch", "cat_id"]
    missing_cols = [c for c in critical if c not in df.columns]

    # 2. Нет пропусков в критичных колонках
    if missing_cols:
        _check(checks, "No nulls in critical columns", False, f"missing columns: {missing_cols}")
    else:
        null_counts = df[critical].isna().sum().to_dict()
        total_nulls = int(sum(null_counts.values()))
        _check(checks, "No nulls in critical columns", total_nulls == 0, f"nulls={total_nulls} {null_counts}")

    # 3. target только 0/1
    if "target" in df.columns:
        bad = ~df["target"].isin([0, 1])
        _check(checks, "target in {0, 1}", not bool(bad.any()), f"invalid={int(bad.sum())}")

    # 4. amount >= 0
    if "amount" in df.columns:
        bad = df["amount"] < 0
        _check(checks, "amount >= 0", not bool(bad.any()), f"invalid={int(bad.sum())}")

    # 5. transaction_time валидно и покрывает ровно 3 месяца
    if "transaction_time" in df.columns:
        ts = pd.to_datetime(df["transaction_time"], errors="coerce")
        invalid_ts = int(ts.isna().sum())
        months = int(ts.dt.to_period("M").nunique()) if invalid_ts < len(ts) else 0
        _check(
            checks,
            "transaction_time valid and covers 3 months",
            invalid_ts == 0 and months == 3,
            f"invalid_ts={invalid_ts}, months={months}",
        )

    # 6. Нет точных дублей строк
    dup_count = int(df.duplicated().sum())
    _check(checks, "No exact duplicate rows", dup_count == 0, f"duplicates={dup_count}")

    # 7. Доля фрода в разумных пределах
    if "target" in df.columns and len(df) > 0:
        rate = float(df["target"].mean())
        _check(checks, "Fraud rate in [0.001, 0.2]", 0.001 <= rate <= 0.2, f"rate={rate:.4f}")

    # 8. Категориальные поля непустые
    for col in ["merch", "gender", "us_state", "jobs"]:
        if col in df.columns:
            empty = int((df[col].astype(str).str.strip() == "").sum())
            _check(checks, f"{col} non-empty", empty == 0, f"empty={empty}")

    DQ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ts_file = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = DQ_REPORTS_DIR / f"dq_report_{ts_file}.html"
    html = _render_html(checks, CURRENT_DATASET, generated_at)
    report_path.write_text(html, encoding="utf-8")
    (DQ_REPORTS_DIR / "latest.html").write_text(html, encoding="utf-8")

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    logger.info("Data Quality: %s. Отчёт: %s", overall, report_path)
    for c in checks:
        logger.info("  %s | %s | %s", c["status"], c["rule"], c["details"])

    return {
        "dataset_path": str(CURRENT_DATASET),
        "report_path": str(report_path),
        "overall": overall,
        "checks": checks,
    }