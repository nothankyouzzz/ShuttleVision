from dataclasses import dataclass


@dataclass(slots=True)
class DetectionBox:
    frame_index: int
    timestamp_s: float
    x_min_px: float
    y_min_px: float
    x_max_px: float
    y_max_px: float
    score: float
    cls: str = "shuttle"

    @property
    def center_u_px(self) -> float:
        return 0.5 * (self.x_min_px + self.x_max_px)

    @property
    def center_v_px(self) -> float:
        return 0.5 * (self.y_min_px + self.y_max_px)


@dataclass(slots=True)
class TrackPoint2D:
    u_px: float
    v_px: float
    timestamp_s: float
    score: float


@dataclass(slots=True)
class Track2D:
    track_id: int
    points: list[TrackPoint2D]
    is_complete: bool = False  # 轨迹是否结束（丢失目标）


@dataclass(slots=True)
class DetectorConfig:
    model_path: str
    conf_threshold: float = 0.3
    nms_threshold: float = 0.5
    # 规则过滤相关
    min_box_area_px2: float = 4.0
    max_box_area_px2: float = 400.0


@dataclass(slots=True)
class TrackerConfig:
    max_missing_frames: int = 5
    max_speed_px_per_s: float = 2000.0
    # 速度/加速度规则阈值等...


class DetectionError(RuntimeError):
    pass


class TrackingError(RuntimeError):
    pass
