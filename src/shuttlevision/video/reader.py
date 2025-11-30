from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import cv2

from .backend import OpenCVBackend, VideoBackend
from .types import DecodedFrame, VideoIOConfig, VideoIOError, VideoMetadata

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _DecodeContext:
    raw_fps: float
    time_per_frame_s: float
    next_raw_index: int = 0
    next_output_index: int = 0
    last_output_time_s: float | None = None


def _should_emit_frame(
    timestamp_s: float,
    last_output_time_s: float | None,
    target_fps: float | None,
) -> bool:
    if target_fps is None:
        return True
    if last_output_time_s is None:
        return True
    return timestamp_s - last_output_time_s >= 1.0 / target_fps


class _FrameIterator:
    """Internal iterator to keep frames() lean and testable."""

    def __init__(
        self,
        backend: VideoBackend,
        metadata: VideoMetadata,
        config: VideoIOConfig,
    ):
        self._backend = backend
        self._metadata = metadata
        self._config = config
        self._ctx = _DecodeContext(
            raw_fps=metadata.fps,
            time_per_frame_s=1.0 / metadata.fps,
        )

    def __iter__(self) -> _FrameIterator:
        return self

    def __next__(self) -> DecodedFrame:
        cfg = self._config
        ctx = self._ctx

        while True:
            ok, frame_bgr = self._backend.read()
            if not ok:
                raise StopIteration
            if frame_bgr is None:
                raise VideoIOError("Backend returned empty frame when ok=True")

            timestamp_s = ctx.next_raw_index * ctx.time_per_frame_s
            ctx.next_raw_index += 1

            if cfg.start_time_s is not None and timestamp_s < cfg.start_time_s:
                continue
            if cfg.end_time_s is not None and timestamp_s > cfg.end_time_s:
                raise StopIteration

            if not _should_emit_frame(
                timestamp_s, ctx.last_output_time_s, cfg.target_fps
            ):
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if cfg.target_width_px and cfg.target_height_px:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (cfg.target_width_px, cfg.target_height_px),
                    interpolation=cv2.INTER_AREA,
                )

            ctx.last_output_time_s = timestamp_s
            frame = DecodedFrame(
                frame_index=ctx.next_output_index,
                timestamp_s=timestamp_s,
                image=frame_rgb,
            )
            if frame.frame_index % 200 == 0:
                logger.debug(
                    "Decoded frame %d at %.3f s (raw idx %d)",
                    frame.frame_index,
                    timestamp_s,
                    ctx.next_raw_index - 1,
                )
            ctx.next_output_index += 1
            return frame


class VideoReader:
    """Opens a video and streams frames according to the provided config."""

    def __init__(
        self,
        path: str,
        config: VideoIOConfig | None = None,
        backend_factory: Callable[[str], VideoBackend] | None = None,
    ):
        self._path: str = path
        self._config: VideoIOConfig = config or VideoIOConfig()
        self._metadata: VideoMetadata | None = None
        self._backend_factory: Callable[[str], VideoBackend] = (
            backend_factory or OpenCVBackend
        )
        self._backend: VideoBackend | None = None

    def open(self) -> None:
        """Open the video source, read metadata, and initialize the backend."""
        self._config.validate()
        backend = self._backend_factory(self._path)
        metadata = backend.open()
        self._metadata = metadata
        self._backend = backend
        logger.debug(
            "Opened video %s (%dx%d, %.2f fps, %d frames)",
            metadata.path,
            metadata.width_px,
            metadata.height_px,
            metadata.fps,
            metadata.num_frames,
        )

    @property
    def metadata(self) -> VideoMetadata:
        """Video metadata, available after open()."""
        if self._metadata is None:
            raise VideoIOError("Video not opened")
        return self._metadata

    def frames(self) -> Iterator[DecodedFrame]:
        """
        Yield decoded frames, handling:
        - time window cropping
        - fps downsampling
        - resolution scaling
        """
        if self._backend is None or self._metadata is None:
            raise VideoIOError("Video not opened")

        return _FrameIterator(self._backend, self._metadata, self._config)

    def close(self) -> None:
        """Release backend resources."""
        if self._backend is not None:
            self._backend.close()
            logger.debug("Released video %s", self._path)
        self._backend = None
        self._metadata = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@contextmanager
def iter_decoded_frames(
    path: str,
    config: VideoIOConfig | None = None,
):
    """Context manager yielding (metadata, frames) while auto-releasing resources.

    Example:

        with iter_decoded_frames(path, cfg) as (metadata, frame_iter):
            for frame in frame_iter:
                ...
    """

    with VideoReader(path, config) as reader:
        yield reader.metadata, reader.frames()


def iter_frames(
    path: str, config: VideoIOConfig | None = None
) -> Iterator[DecodedFrame]:
    """Convenience: iterate frames without caring about metadata."""
    with VideoReader(path, config) as reader:
        yield from reader.frames()
