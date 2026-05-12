"""Label-container public surface re-exported through ``xpkg.model``."""

from xpkg.io.labels.model import Labels, SuggestionFrame
from xpkg.io.labels.video_types import VideoProtocol

__all__ = ["Labels", "SuggestionFrame", "VideoProtocol"]
