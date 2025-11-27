from abc import ABC, abstractmethod
from typing import override

from shuttlevision.video.types import DecodedFrame, VideoMetadata

from .types import DetectionBox, DetectorConfig


class ShuttleDetector(ABC):
    @abstractmethod
    def detect(self, frame: DecodedFrame) -> list[DetectionBox]:
        """
        对单帧进行推理，返回候选检测框（可能为空）。
        """


class YoloShuttleDetector(ShuttleDetector):
    def __init__(self, cfg: DetectorConfig):
        self._cfg: DetectorConfig = cfg
        self._model = self._load_model(cfg.model_path)

    def _load_model(self, path: str):
        # 加载 YOLO/其他模型
        ...

    @override
    def detect(self, frame: DecodedFrame) -> list[DetectionBox]:
        # 1. 将 frame.image 转为模型输入
        # 2. 模型前向
        # 3. NMS + 阈值过滤
        # 4. 调用 rule-based filter
        ...


def filter_detections_by_rules(
    dets: list[DetectionBox],
    frame_meta: VideoMetadata,
    cfg: DetectorConfig,
) -> list[DetectionBox]:
    """根据 bbox 尺寸、长宽比、场内/场外等规则过滤候选。"""
    filtered: list[DetectionBox] = []
    for d in dets:
        w = d.x_max_px - d.x_min_px
        h = d.y_max_px - d.y_min_px
        area = max(w, 0.0) * max(h, 0.0)
        if area < cfg.min_box_area_px2 or area > cfg.max_box_area_px2:
            continue
        # TODO: 长宽比过滤、场外过滤（利用球场 ROI）
        filtered.append(d)
    return filtered
