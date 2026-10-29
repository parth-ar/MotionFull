"""
network/location_fallback.py — Location fallback handler for GNSS outages.

Per project specification:
  - IP-based geolocation is completely disabled.
  - Position holding is completely disabled. If GNSS is unavailable, coordinates
    must show as unavailable (None) rather than reporting stale or false information.
"""

import threading

from config import (
    ENABLE_GPS_FALLBACK,
    GNSS_FALLBACK_TIMEOUT_SEC,
    GNSS_FALLBACK_REFRESH_SEC,
)


def get_laptop_location():
    """Stub: IP-based geolocation is disabled. Always returns None."""
    return None, None, None, None


def gps_fallback_worker(stop_event: threading.Event,
                         enable: bool = False,
                         timeout_sec: float = GNSS_FALLBACK_TIMEOUT_SEC,
                         refresh_sec: float = GNSS_FALLBACK_REFRESH_SEC) -> None:
    """Stub: Position holding is completely disabled.
    If GNSS is disconnected or has no fix, coordinates remain None.
    """
    return
