from collections.abc import Iterator

from shuttlevision.tracking.detector import ShuttleDetector
from shuttlevision.tracking.tracker import MultiObjectTracker
from shuttlevision.tracking.types import Track2D
from shuttlevision.video import DecodedFrame


def run_detection_and_tracking(
    frames: Iterator[DecodedFrame],
    detector: ShuttleDetector,
    tracker: MultiObjectTracker,
) -> list[Track2D]:
    """
    遍历视频帧，完成整段视频的检测 + 跟踪。
    """
    for frame in frames:
        dets = detector.detect(frame)
        tracker.update(frame.frame_index, frame.timestamp_s, dets)

    tracks = tracker.finalize()
    return tracks
