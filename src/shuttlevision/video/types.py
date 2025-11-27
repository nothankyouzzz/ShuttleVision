from dataclasses import dataclass

import numpy as np

Array3U8 = np.ndarray  # shape=(H, W, 3), dtype=uint8


@dataclass(slots=True)
class VideoIOConfig:
    target_width_px: int | None = None
    target_height_px: int | None = None
    target_fps: float | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None


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
