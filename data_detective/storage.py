"""
Data Detective SQLite Persistence Layer.
Stores and retrieves investigation history, analytical snapshots,
hypotheses, contributions, and generated reports.
"""

from __future__ import annotations
import json
import sqlite3
from typing import Dict, List, Optional, Any
from pathlib import Path

from data_detective.schemas import InvestigationRun


DEFAULT_DB_PATH = Path("data_detective.db")


def get_db_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Initializes SQLite schema for storing investigation runs and audit trails.
    """
    with get_db_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                metric_col TEXT NOT NULL,
                date_col TEXT,
                baseline_label TEXT,
                anomaly_label TEXT,
                pct_change REAL,
                direction TEXT,
                detection_method TEXT,
                payload_json TEXT NOT NULL,
                executive_summary TEXT,
                llm_explanation TEXT
            )
        """)
        conn.commit()


def save_investigation(run: InvestigationRun, db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Saves or updates an investigation run in SQLite.
    """
    init_db(db_path)
    run_dict = run.to_dict()
    payload = json.dumps(run_dict)

    with get_db_connection(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO investigations (
                id, title, created_at, dataset_name, row_count,
                metric_col, date_col, baseline_label, anomaly_label,
                pct_change, direction, detection_method,
                payload_json, executive_summary, llm_explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run.id,
            run.title,
            run.created_at,
            run.dataset_name,
            run.row_count,
            run.metric_col,
            run.date_col,
            run.anomaly_window.baseline_period_label,
            run.anomaly_window.anomaly_period_label,
            run.anomaly_window.relative_pct_change,
            run.anomaly_window.direction,
            run.anomaly_window.detection_method,
            payload,
            run.executive_summary,
            run.llm_explanation
        ))
        conn.commit()


def list_investigations(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Lists metadata for all saved investigations, ordered newest first.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("""
            SELECT id, title, created_at, dataset_name, row_count,
                   metric_col, pct_change, direction, detection_method
            FROM investigations
            ORDER BY created_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


def load_investigation(run_id: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Loads complete payload dictionary for a specific investigation.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("SELECT payload_json FROM investigations WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row["payload_json"])
    return None


def delete_investigation(run_id: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """
    Deletes an investigation by ID.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM investigations WHERE id = ?", (run_id,))
        conn.commit()
        return cursor.rowcount > 0
