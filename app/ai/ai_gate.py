"""
AI gateway/facade.
"""

from app.core.logging import get_logger

logger = get_logger(__name__)


class AIGate:
    """Entry point for AI-related integrations."""

    def __init__(self) -> None:
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled


# Global instance
ai_gate = AIGate()
