from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2

from .types import Array3U8, VideoIOError, VideoMetadata


class VideoBackend(ABC):
    """Backend interface to allow swapping OpenCV/PyAV/etc."""

    def __init__(self, path: str):
        self._path = Path(path)

    @abstractmethod
    def open(self) -> VideoMetadata: ...

    @abstractmethod
    def read(self) -> tuple[bool, Array3U8 | None]: ...

    @abstractmethod
    def close(self) -> None: ...


class OpenCVBackend(VideoBackend):
    """OpenCV VideoCapture implementation."""

    def __init__(self, path: str):
        super().__init__(path)
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    def open(self) -> VideoMetadata:
        if self._cap is not None:
            self.close()

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise VideoIOError(f"Cannot open video: {self._path}")

        width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            cap.release()
            raise VideoIOError("Invalid FPS from video")
        if width_px <= 0 or height_px <= 0:
            cap.release()
            raise VideoIOError("Invalid resolution from video")

        duration_s = num_frames / fps if num_frames > 0 else 0.0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_chars = [
            chr((fourcc >> 0) & 0xFF),
            chr((fourcc >> 8) & 0xFF),
            chr((fourcc >> 16) & 0xFF),
            chr((fourcc >> 24) & 0xFF),
        ]
        codec = "".join(codec_chars).strip("\x00") or None

        self._metadata = VideoMetadata(
            path=str(self._path),
            width_px=width_px,
            height_px=height_px,
            fps=fps,
            num_frames=num_frames,
            duration_s=duration_s,
            codec=codec,
        )
        self._cap = cap
        return self._metadata

    def read(self) -> tuple[bool, Array3U8 | None]:
        if self._cap is None:
            raise VideoIOError("Backend not opened")
        ok, frame_bgr = self._cap.read()
        return ok, frame_bgr

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._metadata = None
