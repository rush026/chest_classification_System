# ==========================================================
# src/history.py
# Patient history storage using SQLite
# Stores predictions, vitals, severity scores over time
# ==========================================================

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "patient_history.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_name TEXT,
                age REAL,
                temperature REAL,
                pulse REAL,
                sys_bp REAL,
                dia_bp REAL,
                rr REAL,
                sats REAL,
                clinical_prediction TEXT,
                clinical_probability REAL,
                image_prediction TEXT,
                image_confidence REAL,
                severity_score REAL,
                severity_level TEXT,
                notes TEXT
            )
        """)
        conn.commit()


def save_prediction(
    patient_name: str,
    age: float,
    vitals: dict,
    clinical_result: Optional[dict] = None,
    image_result: Optional[dict] = None,
    severity_score: Optional[float] = None,
    severity_level: Optional[str] = None,
    notes: str = "",
) -> int:
    """Save a prediction record. Returns the new record ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions (
                timestamp, patient_name, age,
                temperature, pulse, sys_bp, dia_bp, rr, sats,
                clinical_prediction, clinical_probability,
                image_prediction, image_confidence,
                severity_score, severity_level, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(),
            patient_name,
            age,
            vitals.get("temperature"),
            vitals.get("pulse"),
            vitals.get("sys"),
            vitals.get("dia"),
            vitals.get("rr"),
            vitals.get("sats"),
            clinical_result.get("prediction") if clinical_result else None,
            clinical_result.get("probability_positive") if clinical_result else None,
            image_result.get("predicted_class") if image_result else None,
            image_result.get("confidence") if image_result else None,
            severity_score,
            severity_level,
            notes,
        ))
        conn.commit()
        return cursor.lastrowid


def get_all_predictions(patient_name: Optional[str] = None) -> List[Dict]:
    """Retrieve all predictions, optionally filtered by patient name."""
    init_db()
    with get_connection() as conn:
        if patient_name:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE patient_name LIKE ? ORDER BY timestamp DESC",
                (f"%{patient_name}%",)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(row) for row in rows]


def get_patient_trend(patient_name: str) -> List[Dict]:
    """Get severity score trend for a specific patient over time."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT timestamp, severity_score, severity_level, clinical_probability, sats "
            "FROM predictions WHERE patient_name = ? ORDER BY timestamp ASC",
            (patient_name,)
        ).fetchall()
        return [dict(row) for row in rows]


def delete_prediction(record_id: int):
    """Delete a prediction record by ID."""
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
        conn.commit()