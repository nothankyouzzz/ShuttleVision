# ShuttleVision 模块 0–3 详细设计（LLD）

本文档基于 ShuttleVision HLD 与 Software-Design-Workflow，对模块 0/1/2/3 进行详细设计。

- 语言：Python 3.11
- 包管理：uv
- 代码组织建议：
  - `shuttlevision/video/`：模块 0（视频 IO）
  - `shuttlevision/geometry/`：模块 1 & 3（相机/球场几何 & 射线）
  - `shuttlevision/tracking/`：模块 2（检测 & 跟踪）
  - `shuttlevision/config/`：通用配置

统一约定：

- 使用类型标注 + `@dataclass(slots=True)`
- 所有真实物理量变量名包含单位后缀：`_m`, `_s`, `_fps`, `_deg` 等
- 日志使用 `logging.getLogger(__name__)`
- 各模块定义自己的异常基类，对外抛业务相关错误类型
- 每个子包维护一个 `types.py` 集中声明本模块的数据类型；跨模块引用通过各子包的 `__init__.py` 导出的符号进行，不直接从其他子包的 `types.py` 导入。

---

### 包内 `types.py` 与 `__init__.py` 规范

- 每个功能子包（如 `shuttlevision.video`, `shuttlevision.geometry`, `shuttlevision.tracking`）都包含一个 `types.py`，集中定义该模块的领域数据类型（dataclass、type alias 等）。
- 子包的其它实现文件（如 `reader.py`, `calibration.py`, `detector.py`）在模块内部通过 `from .types import ...` 引用这些类型。
- 子包的 `__init__.py` 负责选择性地 re-export 对外公开的类型与类，构成该子包的正式公共 API。
- 其他模块在跨包引用时，一律使用 `from shuttlevision.<module> import FooType` 的形式，只依赖 `__init__.py` 中显式导出的符号，而不直接访问其他子包的 `types.py`。

示例：

- `shuttlevision/video/types.py` 定义 `VideoIOConfig`, `VideoMetadata`, `DecodedFrame`；

- `shuttlevision/video/__init__.py`：

  ```python
  from .types import VideoIOConfig, VideoMetadata, DecodedFrame
  from .reader import VideoReader, iter_decoded_frames

  __all__ = [
      "VideoIOConfig",
      "VideoMetadata",
      "DecodedFrame",
      "VideoReader",
      "iter_decoded_frames",
  ]
  ```

- 跨模块使用：

  ```python
  from shuttlevision.video import VideoMetadata, DecodedFrame
  ```

## 模块 0：视频解码与预处理（Video I/O）：视频解码与预处理（Video I/O）

### 0.1 职责

- 将不同格式的视频文件解码为统一格式的逐帧流。
- 提供视频元数据 `VideoMetadata`。
- 支持：时间裁剪、FPS 下采样、分辨率缩放。

### 0.2 核心数据结构

文件：`shuttlevision/video/types.py`

```python
from dataclasses import dataclass
from typing import Tuple
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
    def resolution(self) -> Tuple[int, int]:
        return self.width_px, self.height_px

@dataclass(slots=True)
class DecodedFrame:
    frame_index: int
    timestamp_s: float
    image: Array3U8
```

异常类型：

```python
class VideoIOError(RuntimeError):
    pass
```

### 0.3 公共接口

文件：`shuttlevision/video/reader.py`

```python
from collections.abc import Iterator
from .types import VideoIOConfig, VideoMetadata, DecodedFrame

class VideoReader:
    """
    负责打开视频并按配置解码成标准帧序列。
    """

    def __init__(self, path: str, config: VideoIOConfig | None = None):
        self._path = path
        self._config = config or VideoIOConfig()
        self._metadata: VideoMetadata | None = None
        self._cap = None  # 底层解码器句柄（例如 OpenCV VideoCapture）

    def open(self) -> None:
        """打开视频源，读取基本元数据，初始化解码器。"""
        ...

    @property
    def metadata(self) -> VideoMetadata:
        """解码后可访问的视频元数据。"""
        if self._metadata is None:
            raise VideoIOError("Video not opened")
        return self._metadata

    def frames(self) -> Iterator[DecodedFrame]:
        """
        以生成器方式逐帧输出 DecodedFrame。
        应处理：
        - 时间裁剪
        - fps 下采样
        - 分辨率缩放
        """
        ...

    def close(self) -> None:
        """释放底层资源。"""
        ...

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
```

便捷函数：

```python
from contextlib import contextmanager

@contextmanager
def iter_decoded_frames(
    path: str,
    config: VideoIOConfig | None = None,
) -> tuple[VideoMetadata, Iterator[DecodedFrame]]:
    """以上下文管理器的方式提供 (metadata, frames)。

    用法示例：

        with iter_decoded_frames(path, cfg) as (metadata, frame_iter):
            for frame in frame_iter:
                ...

    这样可以确保底层解码资源在 with 代码块结束时被正确释放。
    """
    with VideoReader(path, config) as reader:
        yield reader.metadata, reader.frames()
```

### 0.4 内部流程（伪代码）

```python
import cv2

def frames(self) -> Iterator[DecodedFrame]:
    cfg = self._config
    cap = self._cap

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if raw_fps <= 0:
        raise VideoIOError("Invalid FPS from video")

    time_per_frame_s = 1.0 / raw_fps

    frame_idx_raw = 0
    frame_idx_out = 0
    last_output_time_s: float | None = None

    while True:
        ok, img_bgr = cap.read()
        if not ok:
            break

        timestamp_s = frame_idx_raw * time_per_frame_s
        frame_idx_raw += 1

        if cfg.start_time_s is not None and timestamp_s < cfg.start_time_s:
            continue
        if cfg.end_time_s is not None and timestamp_s > cfg.end_time_s:
            break

        # fps 下采样
        if cfg.target_fps is not None:
            if last_output_time_s is not None:
                if timestamp_s - last_output_time_s < 1.0 / cfg.target_fps:
                    continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 分辨率缩放
        if cfg.target_width_px and cfg.target_height_px:
            img_rgb = cv2.resize(
                img_rgb,
                (cfg.target_width_px, cfg.target_height_px),
                interpolation=cv2.INTER_AREA,
            )

        last_output_time_s = timestamp_s

        yield DecodedFrame(
            frame_index=frame_idx_out,
            timestamp_s=timestamp_s,
            image=img_rgb,
        )
        frame_idx_out += 1
```

### 0.5 错误处理与日志

- 打不开文件、格式不支持 → 抛 `VideoIOError("Cannot open video ...")`。
- 视频 fps / 分辨率读取失败 → 记录 warning，必要时抛错误。
- 每处理 N 帧记录一次进度（`logger.debug`）。

### 0.6 测试建议

- 正常 MP4 解码，检查 `num_frames * 1/fps ≈ duration_s`。
- 检查 `start_time_s` / `end_time_s` 的时间裁剪是否生效。
- 检查 `target_fps`、`target_width_px` / `target_height_px` 下采样行为是否符合预期。
- 不存在文件 / 损坏视频是否正确抛出 `VideoIOError`。

---

## 模块 1：球场三维坐标系构建（Camera–Court Geometry）

### 1.1 职责

- 表达标准球场尺寸与球场坐标系定义。
- 基于像素标注与标准球场三维坐标，估计相机外参（R, t）与内参（K）。
- 提供标定结果的存储 / 加载与误差评估。

### 1.2 数据结构

文件：`shuttlevision/geometry/types.py`

```python
from dataclasses import dataclass
import numpy as np

Array1F = np.ndarray
Array2F = np.ndarray

@dataclass(slots=True)
class CourtDimensions:
    length_m: float
    width_m: float
    net_height_m: float
    # 后续可添加单打/双打不同线等信息

@dataclass(slots=True)
class CourtCoordinateSystem:
    """球场坐标系约定：
    - 原点：本方左后场角（面向球网时左手边，底线与边线交点）
    - x 轴：沿底线指向对面底线方向（正向指向对面）
    - y 轴：沿左到右
    - z 轴：垂直向上
    """
    origin_description: str = "rear-left corner"
    axes_description: str = "x: length, y: width, z: up"

@dataclass(slots=True)
class CameraIntrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    skew: float = 0.0
    k1: float | None = None
    k2: float | None = None
    p1: float | None = None
    p2: float | None = None
    k3: float | None = None

@dataclass(slots=True)
class CameraPose:
    """世界坐标系即球场坐标系"""
    R_wc: Array2F  # shape=(3, 3)，world->camera 旋转
    t_wc_m: Array1F  # shape=(3,)，world->camera 平移，单位: 米
    intrinsics: CameraIntrinsics

@dataclass(slots=True)
class CalibrationInput:
    """来自标定工具的输入数据。"""
    image_width_px: int
    image_height_px: int
    points_image_px: Array2F  # shape=(N, 2)
    points_world_m: Array2F   # shape=(N, 3)

@dataclass(slots=True)
class CalibrationResult:
    camera_pose: CameraPose
    reprojection_error_px: float
    num_points: int
```

异常类型：

```python
class CalibrationError(RuntimeError):
    pass
```

### 1.3 公共接口

文件：`shuttlevision/geometry/calibration.py`

```python
from .types import (
    CalibrationInput,
    CalibrationResult,
    CameraIntrinsics,
    CameraPose,
)


def estimate_camera_pose(
    calib_input: CalibrationInput,
    initial_intrinsics: CameraIntrinsics | None = None,
) -> CalibrationResult:
    """基于 PnP 估计相机位姿与（可选）内参。"""
    ...


def compute_reprojection_error(
    calib_input: CalibrationInput,
    camera_pose: CameraPose,
) -> float:
    """根据当前位姿和内参计算平均重投影误差。"""
    ...


def save_calibration(result: CalibrationResult, path: str) -> None:
    """以 JSON 或 YAML 保存标定结果。"""
    ...


def load_calibration(path: str) -> CalibrationResult:
    """加载标定结果。"""
    ...
```

### 1.4 算法与内部流程

- 使用 OpenCV `solvePnP`：
  - 输入：`points_world_m` (N×3), `points_image_px` (N×2)；
  - 如果 `initial_intrinsics` 存在，则 K 矩阵固定，仅估计 R, t；
  - 否则可先构造一个近似 pinhole 模型估计 fx, fy, cx, cy。
- 将结果打包为 `CameraPose`：
  - `R_wc`：world→camera 旋转；
  - `t_wc_m`：world→camera 平移；
  - intrinsics：`CameraIntrinsics`。
- 使用 `compute_reprojection_error`：
  - 将 3D 点经 R, t, K 投影到像素平面；
  - 计算与原始标注的 L2 距离平均值（px）。
- 若误差超过阈值（例如 2.0 px），抛出 `CalibrationError` 或记录 warning 视配置而定。

### 1.5 错误处理与日志

- 标注点数 < 4 时：直接抛 `CalibrationError("Not enough points for PnP")`。
- `solvePnP` 返回失败：抛 `CalibrationError`。
- 重投影误差过大：
  - 记录 warning，提供建议（例如检查标注、检查球场尺寸是否正确）。

### 1.6 测试建议

- 构造合成数据：已知相机位姿和球场参数，生成 2D 投影点，再用 `estimate_camera_pose` 恢复，检查 R, t 偏差是否在容忍范围内。
- 在真实数据上验证：
  - 检查误差范围是否与肉眼观察接近。

---

## 模块 2：羽毛球检测与像素轨迹生成

### 2.1 职责

- 从模块 0 的帧流中检测羽毛球候选框（object detection）。
- 使用多目标跟踪生成带 Track ID 的 2D 轨迹。
- 使用规则过滤（尺寸、位置、速度等）剔除明显不合理的候选。

### 2.2 数据结构

文件：`shuttlevision/tracking/types.py`

```python
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
    is_complete: bool = False

@dataclass(slots=True)
class DetectorConfig:
    model_path: str
    conf_threshold: float = 0.3
    nms_threshold: float = 0.5
    min_box_area_px2: float = 4.0
    max_box_area_px2: float = 400.0

@dataclass(slots=True)
class TrackerConfig:
    max_missing_frames: int = 5
    max_speed_px_per_s: float = 2000.0
    # 后续可加入加速度等规则阈值
```

异常类型（示意）：

```python
class DetectionError(RuntimeError):
    pass

class TrackingError(RuntimeError):
    pass
```

### 2.3 检测器接口

文件：`shuttlevision/tracking/detector.py`

```python
from abc import ABC, abstractmethod
from shuttlevision.video import DecodedFrame, VideoMetadata
from .types import DetectionBox, DetectorConfig

class ShuttleDetector(ABC):
    @abstractmethod
    def detect(self, frame: DecodedFrame) -> list[DetectionBox]:
        """对单帧进行推理，返回候选检测框（可能为空）。"""


class YoloShuttleDetector(ShuttleDetector):
    def __init__(self, cfg: DetectorConfig):
        self._cfg = cfg
        self._model = self._load_model(cfg.model_path)

    def _load_model(self, path: str):
        # 加载 YOLO/其他模型
        ...

    def detect(self, frame: DecodedFrame) -> list[DetectionBox]:
        # 1. 将 frame.image 转为模型输入
        # 2. 前向推理
        # 3. NMS + 阈值过滤
        # 4. 规则过滤
        ...
```

规则过滤方法（可放在同文件或 `filters.py`）：

```python
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
```

### 2.4 跟踪器接口

文件：`shuttlevision/tracking/tracker.py`

```python
from dataclasses import dataclass
import numpy as np
from .types import DetectionBox, Track2D, TrackPoint2D, TrackerConfig

Array1F = np.ndarray

@dataclass(slots=True)
class _TrackState:
    track_id: int
    kf_state: Array1F  # 卡尔曼状态向量
    last_timestamp_s: float
    last_score: float
    missing_frames: int
    history: list[TrackPoint2D]


class MultiObjectTracker:
    """简化多目标跟踪（匈牙利 + 卡尔曼滤波）。"""

    def __init__(self, cfg: TrackerConfig):
        self._cfg = cfg
        self._next_track_id = 1
        self._active_tracks: list[_TrackState] = []
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
```

### 2.5 管线封装

文件：`shuttlevision/tracking/pipeline.py`

```python
from collections.abc import Iterator
from shuttlevision.video import DecodedFrame
from .detector import ShuttleDetector
from .tracker import MultiObjectTracker
from .types import Track2D


def run_detection_and_tracking(
    frames: Iterator[DecodedFrame],
    detector: ShuttleDetector,
    tracker: MultiObjectTracker,
) -> list[Track2D]:
    """遍历视频帧，完成整段视频的检测 + 跟踪。"""
    for frame in frames:
        dets = detector.detect(frame)
        tracker.update(frame.frame_index, frame.timestamp_s, dets)

    tracks = tracker.finalize()
    return tracks
```

### 2.6 错误处理与日志

- 模型加载失败 → 抛 `DetectionError("Failed to load model ...")`。
- 推理过程异常：
  - 可按配置选择抛错或跳过该帧；
  - 记录 `logger.error` 并附带帧 index。
- 跟踪过程中出现不合理速度：
  - 可以将该点降权（降低 score）或截断轨迹，记录 `logger.warning`。

### 2.7 测试建议

- 构造简单 synthetic 场景，验证单帧检测输出的格式和基本行为。
- 构造直线运动样本，验证 Track2D 的连续性和 ID 稳定性。
- 构造非常大/非常小 bounding box 测试规则过滤。

---

## 模块 3：像素点到三维射线（Pixel-to-3D Rays）

### 3.1 职责

- 使用模块 1 的相机标定结果，将像素点 $u, v$ 转换为世界坐标系下的三维射线。
- 将模块 2 的轨迹 `Track2D` 转换为 `RayObservation` 序列，为后续 3D 轨迹拟合做准备。

### 3.2 数据结构

文件：`shuttlevision/geometry/rays.py`

```python
from dataclasses import dataclass
import numpy as np
from .types import CameraIntrinsics, CameraPose
from shuttlevision.tracking import Track2D, TrackPoint2D

Array1F = np.ndarray

@dataclass(slots=True)
class Ray3D:
    origin_m: Array1F  # shape=(3,)
    direction: Array1F  # shape=(3,), 已归一化

@dataclass(slots=True)
class RayObservation:
    track_id: int
    frame_index: int
    timestamp_s: float
    ray: Ray3D
    score: float
```

异常类型（示意）：

```python
class GeometryError(RuntimeError):
    pass
```

### 3.3 公共接口

```python
from .types import CameraIntrinsics, CameraPose


def pixel_to_camera_ray(
    u_px: float,
    v_px: float,
    intrinsics: CameraIntrinsics,
) -> Array1F:
    """将像素坐标变换到相机坐标系中的单位方向向量。"""
    ...


def camera_to_world_ray(
    ray_cam: Array1F,
    camera_pose: CameraPose,
) -> Ray3D:
    """将相机坐标系下的射线转换到世界坐标系。"""
    ...


def pixel_to_world_ray(
    u_px: float,
    v_px: float,
    camera_pose: CameraPose,
) -> Ray3D:
    """组合 pixel_to_camera_ray + camera_to_world_ray。"""
    ...


def convert_track_to_rays(
    track: Track2D,
    camera_pose: CameraPose,
) -> list[RayObservation]:
    """将单条 2D 轨迹转为 RayObservation 序列。"""
    ...


def convert_tracks_to_rays(
    tracks: list[Track2D],
    camera_pose: CameraPose,
) -> list[RayObservation]:
    """批量转换所有轨迹。"""
    rays: list[RayObservation] = []
    for t in tracks:
        rays.extend(convert_track_to_rays(t, camera_pose))
    return rays
```

### 3.4 算法细节

`pixel_to_camera_ray`：

```python
import numpy as np


def pixel_to_camera_ray(u_px: float, v_px: float, K: CameraIntrinsics) -> Array1F:
    if K.fx_px == 0 or K.fy_px == 0:
        raise GeometryError("Invalid intrinsics: fx or fy is zero")

    # 归一化坐标
    x_n = (u_px - K.cx_px) / K.fx_px
    y_n = (v_px - K.cy_px) / K.fy_px

    v_cam = np.array([x_n, y_n, 1.0], dtype=float)
    norm = np.linalg.norm(v_cam)
    if norm < 1e-9:
        raise GeometryError("Degenerate ray direction")

    return v_cam / norm
```

`camera_to_world_ray`：

```python

def camera_to_world_ray(ray_cam: Array1F, pose: CameraPose) -> Ray3D:
    R_wc = pose.R_wc
    t_wc_m = pose.t_wc_m

    # R_wc: world->camera，因此 R_cw = R_wc^T
    R_cw = R_wc.T

    direction_world = R_cw @ ray_cam
    direction_world /= np.linalg.norm(direction_world)

    # 相机光心在世界坐标系下的位置：
    # x_cam = R_wc * x_world + t_wc
    # 令 x_cam = 0 => x_world = -R_cw * t_wc
    origin_m = -R_cw @ t_wc_m

    return Ray3D(origin_m=origin_m, direction=direction_world)
```

`pixel_to_world_ray`：

```python

def pixel_to_world_ray(u_px: float, v_px: float, pose: CameraPose) -> Ray3D:
    ray_cam = pixel_to_camera_ray(u_px, v_px, pose.intrinsics)
    return camera_to_world_ray(ray_cam, pose)
```

`convert_track_to_rays`：

```python

def convert_track_to_rays(track: Track2D, pose: CameraPose) -> list[RayObservation]:
    obs: list[RayObservation] = []
    for idx, p in enumerate(track.points):
        ray = pixel_to_world_ray(p.u_px, p.v_px, pose)
        obs.append(
            RayObservation(
                track_id=track.track_id,
                frame_index=idx,  # 或实际 frame_index，视 TrackPoint2D 是否携带
                timestamp_s=p.timestamp_s,
                ray=ray,
                score=p.score,
            )
        )
    return obs
```

### 3.5 错误处理与日志

- 内参非法（`fx_px == 0` 或 `fy_px == 0`）→ 抛 `GeometryError`。
- 射线方向向量长度趋近 0 → 抛 `GeometryError("Degenerate ray direction")`。
- 日志主要用于 debug：
  - 记录部分样本的射线 origin 和 direction；
  - 检查某条轨迹上相邻时刻射线的变化是否平滑。

### 3.6 测试建议

- 简单相机：相机位于原点、朝 z 轴、K 为单位阵：
  - 像素中心 (0, 0) 应映射到 (0, 0, 1) 方向；
  - 四角像素应对应合理方向。
- 与模块 1 联动测试：
  - 取一个世界坐标系下已知点，使用相机位姿投影到像素；
  - 再用 `pixel_to_world_ray` + 与某个已知平面求交，检查反推位置误差。
