"""
frame_manager.py — Threaded frame capture with queue-based producer-consumer.

Reads frames from a camera source (webcam, RTSP, or video file) on a
background thread and exposes them via a queue for the main thread to consume.
"""

import threading
import time
from queue import Queue, Empty
from typing import Optional, Union

import cv2
import numpy as np


class FrameManager:
    """
    Captures video frames on a daemon thread and pushes them to a queue.

    Parameters
    ----------
    source : int | str
        Webcam index (e.g. 0) or RTSP/file path string.
    process_every_n : int
        Only enqueue every Nth frame (frame-skipping for perf).
    queue_size : int
        Max queue depth; old frames are dropped to keep latency low.
    resolution : tuple[int, int]
        Target (width, height) to resize frames before enqueuing.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        process_every_n: int = 1,
        queue_size: int = 2,
        resolution: tuple = (640, 480),
    ):
        self.source = source
        self.process_every_n = max(1, process_every_n)
        self.resolution = resolution
        self._queue: Queue = Queue(maxsize=queue_size)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_count = 0
        self._fps: float = 0.0

    # ── lifecycle ────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the video source and start the capture thread. Returns True on success."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Signal the capture thread to stop and release resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── public API ───────────────────────────────────────────────

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Return the latest frame or None if the queue is empty after *timeout*.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._running

    # ── private capture loop ─────────────────────────────────────

    def _capture_loop(self):
        """Read frames continuously and push every Nth frame to the queue."""
        count = 0
        prev_time = time.time()

        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                # For video files: reached end
                self._running = False
                break

            count += 1
            if count % self.process_every_n != 0:
                continue

            # Resize
            frame = cv2.resize(frame, self.resolution)

            # Drop old frame if queue is full (keep only latest)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame)

            # Compute effective FPS
            now = time.time()
            elapsed = now - prev_time
            if elapsed > 0:
                self._fps = 1.0 / elapsed
            prev_time = now

        self._running = False
