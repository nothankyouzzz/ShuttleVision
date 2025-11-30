from dataclasses import dataclass

import numpy as np
from jaxtyping import UInt8

Array3U8 = UInt8[np.ndarray, "height width 3"]  # RGB frame


@dataclass(slots=True)
class VideoIOConfig:
    """Options for video decoding.

    Time windowing uses a half-open interval [start_time_s, end_time_s): frames at
    start_time_s are included, frames at end_time_s are excluded.
    """
    target_width_px: int | None = None
    target_height_px: int | None = None
    target_fps: float | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None

    def validate(self) -> None:
        if self.target_fps is not None and self.target_fps <= 0:
            raise VideoIOError("target_fps must be positive")
        if (
            self.start_time_s is not None
            and self.end_time_s is not None
            and self.start_time_s >= self.end_time_s
        ):
            raise VideoIOError("start_time_s must be earlier than end_time_s")


@dataclass(slots=True)
class VideoMetadata:
    path: str
    width_px: int
    height_px: int
    fps: float
    num_frames: int
    duration_s: float
    codec: str | None = None

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width_px, self.height_px


@dataclass(slots=True)
class DecodedFrame:
    frame_index: int
    timestamp_s: float
    image: Array3U8


class VideoIOError(RuntimeError):
    pass
