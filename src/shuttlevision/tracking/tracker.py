from dataclasses import dataclass

from shuttlevision.geometry.types import Array1F

from .types import DetectionBox, Track2D, TrackerConfig, TrackPoint2D


class MultiObjectTracker:
    """简化多目标跟踪（匈牙利 + 卡尔曼滤波）。"""

    def __init__(self, cfg: TrackerConfig):
        self._cfg = cfg
        self._next_track_id = 1
        self._active_tracks: list[TrackState] = []
        self._finished_tracks: list[Track2D] = []

    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        detections: list[DetectionBox],
    ) -> list[Track2D]:
        """输入当前帧检测结果，内部更新所有轨迹状态。"""
        # 1. 预测步骤（对每条轨迹使用卡尔曼预测）
        # 2. 构造代价矩阵（欧氏距离、速度约束等）
        # 3. 匈牙利算法匹配轨迹与检测
        # 4. 更新/新建/终止轨迹
        ...

    def finalize(self) -> list[Track2D]:
        """视频结束时调用，输出所有轨迹。"""
        # 将 active_tracks 全部收尾到 finished_tracks
        ...
        return self._finished_tracks


@dataclass(slots=True)
class TrackState:
    track_id: int
    kf_state: Array1F  # 卡尔曼状态向量
    last_timestamp_s: float
    last_score: float
    missing_frames: int
    history: list[TrackPoint2D]
