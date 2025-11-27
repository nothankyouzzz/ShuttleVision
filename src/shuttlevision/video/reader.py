from collections.abc import Iterator
from contextlib import contextmanager

from .types import DecodedFrame, VideoIOConfig, VideoIOError, VideoMetadata


class VideoReader:
    """
    负责打开视频并按配置解码成标准帧序列。
    """

    def __init__(self, path: str, config: VideoIOConfig | None = None):
        self._path: str = path
        self._config: VideoIOConfig | None = config or VideoIOConfig()
        self._metadata: VideoMetadata | None = None
        self._cap: str | None = None  # 底层解码器句柄（例如 OpenCV VideoCapture）

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


@contextmanager
def iter_decoded_frames(
    path: str,
    config: VideoIOConfig | None = None,
):
    """以上下文管理器的方式提供 (metadata, frames)。

    用法示例：

        with iter_decoded_frames(path, cfg) as (metadata, frame_iter):
            for frame in frame_iter:
                ...

    这样可以确保底层解码资源在 with 代码块结束时被正确释放。
    """

    with VideoReader(path, config) as reader:
        yield reader.metadata, reader.frames()
