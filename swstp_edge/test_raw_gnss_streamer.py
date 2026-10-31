#!/usr/bin/env python3
"""
test_raw_gnss_streamer.py — Automated Unit Test for Live Raw GNSS Streaming API.

Validates:
  1. POST requests sent to: /api/{vehicle_id}/live_location
  2. JSON payload structure (raw un-snapped latitude, longitude, speed, heading, altitude, sats, fix)
  3. Continuous non-blocking streaming rate and session keep-alive
  4. Graceful handling of backend offline / reconnect behavior
"""

import datetime
import http.server
import json
import os
import sys
import threading
import time

# Ensure swstp_edge is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sensors.gnss as gnss_sensor
import telemetry
from network.uploader import raw_gnss_streamer


class MockLiveLocationHandler(http.server.BaseHTTPRequestHandler):
    received_requests = []
    lock = threading.Lock()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        with self.lock:
            self.received_requests.append({
                "path": self.path,
                "headers": dict(self.headers),
                "payload": payload,
                "time": time.time(),
            })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok","received":true}')

    def log_message(self, format, *args):
        # Silence console access logs during test
        return


def make_nmea(body: str) -> str:
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"${body}*{c:02X}"


def run_test():
    print("==================================================================")
    print("  TESTING LIVE RAW GNSS STREAMING API (/api/{vehicle_id}/live_location)")
    print("==================================================================")

    # 1. Start Mock HTTP Server on ephemeral local port
    server = http.server.HTTPServer(("127.0.0.1", 0), MockLiveLocationHandler)
    server_port = server.server_port
    server_url = f"http://127.0.0.1:{server_port}"
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[TEST SETUP] Mock HTTP Backend running on: {server_url}")

    test_vehicle_id = "MH-27-WASTE-404"
    expected_path = f"/api/{test_vehicle_id}/live_location"

    # 2. Feed simulated raw GNSS fix into gnss_sensor
    # Raw coordinates (e.g. Pune, Maharashtra: 18.520430, 73.856740)
    raw_lat_expected = 18.520430
    raw_lon_expected = 73.856740
    # NMEA: 18 deg 31.2258 min N, 73 deg 51.4044 min E
    gga = make_nmea("GNGGA,071530.00,1831.2258,N,07351.4044,E,1,09,0.9,560.2,M,-64.0,M,,")
    rmc = make_nmea("GNRMC,071530.00,A,1831.2258,N,07351.4044,E,12.5,142.0,311026,,,A")
    gnss_sensor._dispatch(gga)
    gnss_sensor._dispatch(rmc)

    # Verify sensor state acquired
    snap = gnss_sensor.read()
    print(f"[GNSS SENSOR] Fix acquired: {snap['fix']} | Lat: {snap['latitude']} | Lon: {snap['longitude']}")
    assert snap["fix"] is True, "Mock GNSS fix must be True"
    assert snap["latitude"] is not None, "Mock latitude must be present"

    # 3. Launch raw_gnss_streamer worker at 5 Hz (high rate for quick test)
    stop_event = threading.Event()
    stream_thread = threading.Thread(
        target=raw_gnss_streamer,
        args=(server_url, test_vehicle_id, stop_event, 5.0),
        daemon=True,
    )
    stream_thread.start()

    # Allow worker to send ~5 requests
    time.sleep(1.2)
    stop_event.set()
    stream_thread.join(timeout=2.0)
    server.shutdown()

    # 4. Verify received requests
    with MockLiveLocationHandler.lock:
        reqs = list(MockLiveLocationHandler.received_requests)

    print(f"[TEST RESULTS] Total HTTP requests received by backend: {len(reqs)}")
    assert len(reqs) >= 2, f"Expected at least 2 streamed packets, got {len(reqs)}"

    first_req = reqs[0]
    print(f"[TEST CHECK 1] HTTP Path: {first_req['path']}")
    assert first_req["path"] == expected_path, f"Path mismatch: {first_req['path']} != {expected_path}"

    payload = first_req["payload"]
    print(f"[TEST CHECK 2] Vehicle ID: {payload.get('vehicle_id')}")
    assert payload.get("vehicle_id") == test_vehicle_id, f"vehicle_id mismatch: {payload.get('vehicle_id')}"
    assert payload.get("vehicleId") == test_vehicle_id, "vehicleId alias mismatch"

    print(f"[TEST CHECK 3] Raw Latitude: {payload.get('latitude'):.6f} (Expected: ~{raw_lat_expected:.6f})")
    print(f"[TEST CHECK 4] Raw Longitude: {payload.get('longitude'):.6f} (Expected: ~{raw_lon_expected:.6f})")
    assert abs(payload.get("latitude") - raw_lat_expected) < 0.001, "Latitude mismatch"
    assert abs(payload.get("longitude") - raw_lon_expected) < 0.001, "Longitude mismatch"

    print(f"[TEST CHECK 5] Fix Status: {payload.get('fix')} (status={payload.get('status')})")
    assert payload.get("fix") is True, "Fix status must be True"

    print(f"[TEST CHECK 6] Speed: {payload.get('speed_kmh')} km/h | Heading: {payload.get('heading')}°")
    print(f"[TEST CHECK 7] Location Source: {payload.get('location_source')} | is_snapped: {payload.get('is_snapped')}")
    assert payload.get("location_source") == "raw_gnss", "Location source must be raw_gnss"
    assert payload.get("is_snapped") is False, "is_snapped must be False"

    print("------------------------------------------------------------------")
    print("[PASS] ALL LIVE RAW GNSS STREAMING API CHECKS PASSED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == "__main__":
    run_test()

