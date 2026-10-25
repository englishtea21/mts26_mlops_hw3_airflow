"""DAG проверки качества датасета обучения."""

from datetime import timedelta

import pendulum
from airflow.decorators import dag, task

from pipeline.dq.check_dataset import run_data_quality


@dag(
    dag_id="data_quality",
    description="Проверяет текущий датасет обучения на Data Quality требования и публикует HTML-отчёт",
    schedule="*/5 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Moscow"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "data-quality"],
    doc_md="""
### Data Quality датасета обучения

Проверяет `current.parquet` (текущее окно обучения) по набору правил DQ
и публикует HTML-отчёт в `/opt/airflow/dq_reports`.

Правила:
1. Датасет не пустой.
2. Нет пропусков в критичных колонках (transaction_time, target, amount, merch, cat_id).
3. target принимает только значения 0/1.
4. amount >= 0.
5. transaction_time валиден и покрывает ровно 3 календарных месяца.
6. Нет точных дублей строк.
7. Доля фрода в диапазоне [0.001, 0.2].
8. Категориальные поля (merch, gender, us_state, jobs) непустые.
    """,
)
def data_quality():
    task(run_data_quality, task_id="run_data_quality")()


data_quality()