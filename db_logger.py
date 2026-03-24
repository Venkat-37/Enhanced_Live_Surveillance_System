"""
db_logger.py — SQLite-backed detection logger with analytics queries.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Tuple, Optional


DB_PATH = os.path.join(os.path.dirname(__file__), "detections.db")


class DBLogger:
    """
    Manages a local SQLite database for detection events.

    Schema
    ------
    detections:
        id            INTEGER PRIMARY KEY
        timestamp     TEXT     (ISO-8601)
        zone_name     TEXT
        detected_class TEXT
        confidence    REAL
        snapshot_path TEXT
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ── lifecycle ────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    zone_name       TEXT,
                    detected_class  TEXT    NOT NULL,
                    confidence      REAL    NOT NULL,
                    snapshot_path   TEXT
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── write ────────────────────────────────────────────────────

    def log_detection(
        self,
        zone_name: str,
        detected_class: str,
        confidence: float,
        snapshot_path: Optional[str] = None,
    ):
        """Insert a new detection record."""
        ts = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO detections
                   (timestamp, zone_name, detected_class, confidence, snapshot_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (ts, zone_name, detected_class, confidence, snapshot_path),
            )

    # ── read ─────────────────────────────────────────────────────

    def get_recent(self, limit: int = 50) -> List[dict]:
        """Return the most recent *limit* detections as dicts."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_hourly_counts(self, hours: int = 24) -> List[Tuple[str, int]]:
        """
        Return (hour_label, count) tuples for the last *hours* hours.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT strftime('%Y-%m-%d %H:00', timestamp) AS hour,
                       COUNT(*) AS cnt
                FROM   detections
                WHERE  timestamp >= datetime('now', ?)
                GROUP BY hour
                ORDER BY hour
            """, (f"-{hours} hours",)).fetchall()
        return rows

    def get_zone_counts(self) -> List[Tuple[str, int]]:
        """Return (zone_name, count) for all zones."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT zone_name, COUNT(*) AS cnt
                FROM   detections
                WHERE  zone_name IS NOT NULL
                GROUP BY zone_name
                ORDER BY cnt DESC
            """).fetchall()
        return rows

    def get_class_counts(self) -> List[Tuple[str, int]]:
        """Return (class_name, count) for all detected classes."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT detected_class, COUNT(*) AS cnt
                FROM   detections
                GROUP BY detected_class
                ORDER BY cnt DESC
            """).fetchall()
        return rows

    def get_total_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM detections").fetchone()
        return row[0] if row else 0

    def clear_all(self):
        """Delete all records (useful during testing)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM detections")
