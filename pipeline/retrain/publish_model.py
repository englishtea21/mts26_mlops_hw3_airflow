"""Публикация бинарника модели и фиксация нового окна обучения."""

import json
import logging
import os
from datetime import date
from pathlib import Path

from pipeline.db import advance_training_state, get_conn
from pipeline.settings import CURRENT_DATASET, MODELS_DIR

logger = logging.getLogger(__name__)


def _comparable_scores(train_info: dict):
    """Возвращает (имя_метрики, значение_новой, значение_предыдущей) или None.

    Сравниваем на одной и той же метрике: ROC-AUC, если доступна у обеих
    моделей, иначе F1. Если базовой метрики нет (первая публикация),
    возвращаем None — публикуем без сравнения.
    """
    new_roc = train_info.get("roc_auc")
    new_f1 = train_info.get("f1")
    prev_roc = train_info.get("prev_roc_auc")
    prev_f1 = train_info.get("prev_f1")

    if new_roc is not None and prev_roc is not None:
        return "roc_auc", float(new_roc), float(prev_roc)
    if new_f1 is not None and prev_f1 is not None:
        return "f1", float(new_f1), float(prev_f1)
    return None


def publish_model(train_info: dict) -> str:
    """Публикует модель (обновляет latest.txt) только если качество не хуже базового.

    Окно обучения и training_state сдвигаются всегда — иначе пайплайн застрянет
    на одном месяце и повторит ту же, уже отклонённую, модель. Бинарник и
    метрики сохраняются в любом случае (для аудита), а latest.txt обновляется
    только при улучшении/равенстве качества.
    Возвращает id опубликованной версии.
    """
    version = train_info["version"]
    staging_model = Path(train_info["staging_model"])
    final_model = MODELS_DIR / f"my_catboost_{version}.cbm"
    metrics_path = MODELS_DIR / f"my_catboost_{version}.metrics.json"

    if staging_model.exists():
        os.replace(staging_model, final_model)
    elif not final_model.exists():
        raise FileNotFoundError(f"Нет файла модели для публикации: {staging_model}")

    # Решение о публикации.
    comparison = _comparable_scores(train_info)
    if comparison is None:
        should_publish = True
        metric_name = None
        new_val = None
        prev_val = None
        logger.info(
            "Базовой метрики нет (первая публикация или нет предыдущей модели): "
            "публикуем %s",
            version,
        )
    else:
        metric_name, new_val, prev_val = comparison
        should_publish = new_val >= prev_val
        if should_publish:
            logger.info(
                "Качество не хуже базового (%s: %.4f >= %.4f): публикуем %s",
                metric_name,
                new_val,
                prev_val,
                version,
            )
        else:
            logger.warning(
                "Качество хуже базового (%s: %.4f < %.4f): latest.txt не обновляем, "
                "модель %s остаётся неактивной",
                metric_name,
                new_val,
                prev_val,
                version,
            )

    # Safety: если активной модели нет (latest.txt отсутствует), публикуем всегда.
    if not (MODELS_DIR / "latest.txt").exists():
        if not should_publish:
            logger.warning("latest.txt отсутствует: принудительно публикуем %s", version)
        should_publish = True

    metrics_path.write_text(
        json.dumps(
            {
                "version": version,
                "batch_month": train_info["batch_month"],
                "train_months": train_info["train_months"],
                "test_month": train_info["test_month"],
                "train_rows": train_info["train_rows"],
                "test_rows": train_info["test_rows"],
                "roc_auc": train_info["roc_auc"],
                "f1": train_info["f1"],
                "prev_version": train_info.get("prev_version"),
                "prev_roc_auc": train_info.get("prev_roc_auc"),
                "prev_f1": train_info.get("prev_f1"),
                "comparison_metric": metric_name,
                "published": should_publish,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if should_publish:
        latest_tmp = MODELS_DIR / f".latest.{version}.txt"
        latest_tmp.write_text(version + "\n", encoding="utf-8")
        os.replace(latest_tmp, MODELS_DIR / "latest.txt")
        logger.info("Опубликована модель %s (%s)", version, final_model)
    else:
        logger.info("latest.txt оставлен без изменений")

    pending_window = Path(train_info["dataset_path"])
    if pending_window.exists():
        os.replace(pending_window, CURRENT_DATASET)
        logger.info("Окно обучения обновлено: %s", CURRENT_DATASET)
    elif not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет нового окна обучения: {pending_window}")
    else:
        logger.info("Окно уже перенесено в %s, повторно не сдвигаем.", CURRENT_DATASET)

    batch_month = date.fromisoformat(train_info["batch_month"])
    conn = get_conn()
    try:
        advance_training_state(conn, batch_month)
    finally:
        conn.close()
    return version
