#!/usr/bin/env python3
"""
test_neo6m_gnss.py — Backward-Compatible Alias for test_max_m10s_gnss.py.

NOTE: This test script has been retitled to test_max_m10s_gnss.py for the
u-blox MAX-M10S GNSS module (which shares identical connections with NEO-6M).

Running this file automatically executes test_max_m10s_gnss.py.
Recommended usage:
    python test_max_m10s_gnss.py
    python test_max_m10s_gnss.py --port /dev/serial0 --baud 9600
"""

from test_max_m10s_gnss import (
    main,
    MAXM10SState,
    NEO6MState,
    auto_detect_gnss_port,
    auto_detect_neo6m_port,
    render_dashboard,
    print_stream_line,
    haversine_distance,
    dec_to_dms,
    degrees_to_cardinal,
    rate_hdop,
)

if __name__ == "__main__":
    main()
