"""
snapshot_saver.py — Saves annotated detection frames as JPEG snapshots.
"""

import os
import glob
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np


SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


class SnapshotSaver:
    """
    Saves annotated frames to disk and lists recent snapshots.

    Parameters
    ----------
    output_dir : str
        Directory to save snapshot images.
    """

    def __init__(self, output_dir: str = SNAPSHOTS_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, frame: np.ndarray) -> str:
        """
        Save *frame* as a timestamped JPEG and return the file path.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"snapshot_{ts}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def get_recent(self, n: int = 20) -> List[str]:
        """Return absolute paths to the last *n* snapshots (newest first)."""
        pattern = os.path.join(self.output_dir, "snapshot_*.jpg")
        files = sorted(glob.glob(pattern), reverse=True)
        return files[:n]

    def cleanup(self, keep: int = 200):
        """Delete old snapshots, keeping only the most recent *keep*."""
        pattern = os.path.join(self.output_dir, "snapshot_*.jpg")
        files = sorted(glob.glob(pattern), reverse=True)
        for f in files[keep:]:
            try:
                os.remove(f)
            except OSError:
                pass
