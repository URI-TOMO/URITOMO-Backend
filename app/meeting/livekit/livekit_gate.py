"""
LiveKit gateway/facade.
"""

from app.core.logging import get_logger

logger = get_logger(__name__)


class LiveKitGate:
    """Entry point for LiveKit-related integrations."""

    def __init__(self) -> None:
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled


# Global instance
livekit_gate = LiveKitGate()
