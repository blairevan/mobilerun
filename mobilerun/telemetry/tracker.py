"""Compatibility telemetry API with collection permanently disabled in this fork."""

import logging

from mobilerun.telemetry.events import TelemetryEvent

logger = logging.getLogger("mobilerun-telemetry")


def is_telemetry_enabled(*, config_enabled: bool = False) -> bool:
    """Return false regardless of legacy configuration or environment variables."""
    return False


def print_telemetry_message(*, config_enabled: bool = False) -> None:
    """Report the fixed policy without inspecting or logging event content."""
    logger.debug("Telemetry is permanently disabled by local-only policy.")


def get_user_id() -> str:
    """Return a non-identifying constant without reading or writing user files."""
    return "local"


def capture(
    event: TelemetryEvent,
    user_id: str | None = None,
    *,
    config_enabled: bool = False,
) -> None:
    """Discard events without serialization, logging, storage or network access."""
    return None


async def flush(*, config_enabled: bool = False) -> None:
    """Retain the async compatibility API without creating a telemetry client."""
    return None
