# Pending Changes

Running log of known issues and changes still to be made. Move items to **Done** (with date / commit) once fixed.

## Open

### Evidence upload packets (`POST /api/evidence/upload`)

Context: motion and litter images share one uploader — `_upload_and_cleanup_evidence()` in
`swstp_edge/network/uploader.py`. Motion queue items are built in `swstp_edge/main.py` (~L1135),
litter queue items in `swstp_edge/litter_engine.py` (`_save_worker`, ~L136).

_Items 1, 2, 5, 6 and 7 are done (see **Done**). Items 3 and 4 were moved into item 13._

### Telemetry / streaming packets

_Items 8–12 are done or closed (see **Done** / **Closed**)._

- **Backend follow-up for item 11:** the backend must add `GET /api/camera/{deviceId}/stream-status`
  → `{"active": true|false}` (spec: `swstp_edge/prompts/SITE_LIVE_STREAM_PROMPT.txt`). Until it
  exists, the live feed stays **off** while `STREAM_ON_DEMAND = True`. Set it to `False` in
  `config.py` to stream continuously in the meantime.

### Cleanup / optimization _(on hold)_

13. **[Medium] Clean up every packet sent to the backend.**
    Audit each outbound payload: evidence upload (motion + litter, live + backlog replay), slim
    telemetry, fallback heartbeat, raw GNSS stream, live frame stream, and auth/session calls.
    - Define one schema per packet type (field names, types, units, required vs. optional) in one
      place, and build every payload through it instead of hand-building dicts inline.
    - Use consistent key naming (no snake_case / camelCase mix), consistent enums (`fix`,
      `detectionType`), consistent timestamp format with a UTC offset, and consistent booleans.
    - Remove dead, duplicate and always-default fields. Stop computing data that is never sent
      (e.g. the unsent `rawJson` contents).
    - Get rid of `or`-chains that hide real values; send `None` deliberately or omit it, the same way
      everywhere.
    - Make live and backlog/offline paths produce identical payloads for the same event.
    - Add tests that check each packet type against its schema.
    - Start with item 15 (duplicate fields). It is the biggest size saving.
    - **13a (was item 3). Litter metadata fields are defaults, not real values.**
      - `width` / `height` are sent as `1280` / `720`, but the saved image is the frame plus a banner
        stacked below it (`np.vstack`), so the real height is larger.
      - `compressionQuality` is sent as `80`, but litter JPEGs are encoded at quality `90`.
      - `speedKph` is always `0.0` (the item has no `speed_kph`; the sidecar has `gnss.speed_kmh`).
      - `collectionEventId` is always `0`.
      - Populate these from the actual frame and GNSS data, or drop the ones the backend doesn't need.
    - **13b (was item 4). `isLeftSide` missing on live litter uploads.**
      `is_left_side` is in the litter sidecar but not in the live `upload_item`, so it's only sent
      when the image is replayed from the backlog. Decide together with item 15, which proposes
      dropping `isLeftSide` because it can be derived from `dumpingSide`.

14. **[Medium] Clean up and optimize all sensor and standalone modules.**
    Scope: `sensors/` (`gnss.py`, `imu.py`, `leds.py`, `power.py`, `rtc.py`, `rtc_sync.py`) and the
    separate modules (`camera.py`, `motion.py`, `detector.py`, `litter_engine.py`, `stop_detector.py`,
    `geofence.py`, `roi_side_classifier.py`, `telemetry.py`, `session_logger.py`,
    `network/location_fallback.py`, `config.py`).
    - Give each sensor a common interface (init / read / health / close) with clear error handling and
      reconnect behaviour, and no silent `except: pass`.
    - Remove dead code, unused imports and leftover debug prints. Move magic numbers into `config.py`.
    - Reduce CPU and I/O load: check polling rates, busy loops, redundant reads, frame copies and
      per-loop allocations, and lock contention between threads.
    - Make thread lifecycles and shutdown clean (no orphaned threads, files or serial ports).
    - Make logging consistent (levels, no log spam in the hot loop).
    - Add type hints and docstrings where they are missing. Keep the existing tests green and add
      tests where coverage is thin.
    - **Progress (file by file):**
      - [x] `sensors/imu.py` (2026-11-19): re-init now runs in a background thread
        (`imu-recovery`), so `read()` never blocks the 20 Hz loop. A failed re-init no longer
        disables the IMU permanently: it retries every 5 s. An IMU missing at boot is detected and
        calibrated when it appears. Failure logs are rate-limited (first, then every 60 s). Bench
        machines without `smbus2` never retry. The quaternion was kept (used by
        `test_all_sensors.py`). Tests: `tests/test_imu_recovery.py` (5).
      - [x] `sensors/imu.py` calibration (2026-11-19): options 1–4. The boot calibration is
        accepted only if the vehicle is still (gyro/accel spread limits; a steady turn is checked
        against the trusted baseline). The last good offsets are saved to `imu_calibration.json`
        (git-ignored) and loaded at boot. Re-calibration runs at GNSS-confirmed stops (≥ 5 s,
        ≤ 1 km/h, at most every 10 min, blended 30% into a trusted baseline). Impossible offsets are
        rejected. With no usable offsets the IMU reports invalid (no bad baseline). All limits are in
        `config.py`; tune them from the logged `spread` lines. Also fixed stale IMU data:
        `latest_sensor["imu"]` now always holds the latest result, so a dead IMU reports
        `imu_ok: false`.
      - [x] `sensors/gnss.py` G1 + G2 (2026-11-19). **Confirmed on the Pi: the legacy UBX-CFG-RATE
        is ignored (GGA measured at 1.0/s), so the whole pipeline was running at 1 Hz.** The rate is
        now set with UBX-CFG-VALSET (`CFG-RATE-MEAS`, RAM layer only) and the ACK is checked.
        Fallback order: VALSET rate+filter, then VALSET rate only, then legacy CFG-RATE. The firmware
        is logged (MON-VER), and the real rate is measured and logged 5 s after every connect. The
        sentence filter (`GNSS_NMEA_FILTER`, default on) sends GGA+RMC every epoch and GSV at 1 Hz,
        with VTG/GLL/GSA off. The rate is set by `GNSS_RATE_HZ`. Tests: `tests/test_gnss_config.py`
        (9).
        - **Needs a Pi run:** the startup log must show `ACK via VALSET` and `Measured output rate:
          5.0 Hz ✔`.
      - [x] `sensors/gnss.py` G3–G7 + redundancy scan (2026-11-19). Parsing at 5 Hz now costs
        about 0.34 ms/s, down from 2.9 ms/s (measured on the dev PC):
        - G3: GSV gives only the satellites-in-view count (message 1 of each sequence). Per-satellite
          records and GSA only run with `GNSS_SATELLITE_DETAIL` (set by `test_all_sensors.py`).
        - G4: continuous serial errors on a port that still exists cause a reopen after
          `GNSS_SERIAL_ERROR_RECONNECT_SEC` (2 s), instead of a silent 10 ms loop.
        - G5: the sentence type is checked before the checksum. VTG/GLL/TXT and GSV messages 2+ are
          skipped without an XOR pass.
        - Redundant parsing removed: RMC no longer re-parses position (GGA alone is authoritative),
          and the VTG parser was deleted (it duplicated RMC speed/course).
        - G6: `_target_baud`, `primary_module`, constant `_state` keys, `GNSS_AUTO_DETECT` and
          `GNSS_DETAIL_INTERVAL_SEC` removed. `GNSS_RESET_PIN` is marked as a wiring reference.
        - G7: the 1 Hz / 10 s comments in `telemetry.py` and `geofence.py` were corrected.
        - Tests: `tests/test_gnss_parsing.py` (7).
      - [ ] GNSS follow-ups (suggested, not started):
        - **Legacy README:** it still documents NEO-6M + gpsd + gpsdclient. A running gpsd competes
          for `/dev/serial0`. Check with `systemctl is-active gpsd` on the Pi, and rewrite the README
          section.
        - Event-driven consumers: an epoch counter or event, so the raw GNSS stream sends each fix
          exactly once and telemetry/heading process only new fixes.
        - `CFG-NAVSPG-DYNMODEL` = automotive (less position wander at stops).
        - UBX-NAV-PVT binary instead of NMEA (one struct unpack per epoch, plus hAcc/sAcc accuracy
          and UTC time).
        - GNSS UTC time as an offline RTC sync source (for the `rtc_sync.py` audit).
      - [ ] `sensors/leds.py`: about 120 GPIO writes/s even when nothing changes. Write only on
        change.
      - [ ] `sensors/power.py`, `sensors/rtc.py`, `sensors/rtc_sync.py`: not audited yet.
      - [ ] Telemetry path (not sensor files): heading window fed at 20 Hz (C), heading snaps to 0°
        when stopped (D, needs a hardware check), heartbeat re-reads sensors from a second thread
        (F), unused `rawJson` (G), stop/move log spam (I), silent excepts (J).

15. **[High] Packets repeat the same data under several keys.**
    The same value is sent two or three times under different names (snake_case, camelCase and short
    aliases), which bloats every packet. Pick one name per field, agree it with the backend, and
    remove the rest.
    - **Raw GNSS stream** (`raw_gnss_streamer()`, `uploader.py` ~L541). Sent at 5 Hz, so this is the
      biggest win. 25 keys hold about 14 values:
      - `vehicle_id` / `vehicleId`
      - `device_id` / `deviceId`
      - `latitude` / `lat`
      - `longitude` / `lon` / `lng` (three copies)
      - `altitude` / `altitude_m`
      - `speed_kmh` / `speed`
      - `heading` / `course_deg`
      - `timestamp` (RTC) / `utc_timestamp` / `epoch_ms`: three timestamps for one moment; keep one
        or two.
      - `satellites` / `satellites_in_view`: the second falls back to the first when missing.
      - `location_source: "raw_gnss"` and `is_snapped: False` are constants on this endpoint and add
        nothing per packet.
      - `vehicle_id` is also in the URL path (`/api/{vehicle_id}/live_location`).
    - **Slim telemetry** (`build_slim_packet()`, `telemetry.py` ~L520, and the fallback heartbeat,
      `uploader.py` ~L448): sends both `dumping_side` and `dumpingSide`. Also:
      - `uptime` equals `epoch` (see item 8).
      - `snapped` is always `False`. Road-snapping was removed from the code (`telemetry.py` ~L176),
        and `is_snapped` is `False` on every path. The value is accurate, so this is not a bug as
        first reported, but the field is dead weight. The same applies to `is_snapped` and the
        constant `location_source` in the raw GNSS stream.
      - Telemetry goes out in batches of up to 50 packets per POST, so every extra key is repeated up
        to 50 times per request.
    - **Evidence upload** (`_upload_and_cleanup_evidence()`, `uploader.py` ~L684): sends both
      `dumpingSide` and `dumping_side`. `isLeftSide` can be worked out from `dumpingSide`; keep one.
    - **Internal full packet** (`build_telemetry_packet()`, `telemetry.py` ~L375): `displayLat` /
      `displayLon` copy `lat` / `lon`, and `rawJson` serialises everything again. These aren't sent,
      but they cost CPU and memory 20 times a second. Remove them, or build them only when debugging.
    - Before removing a key, check which name the backend actually reads (or have it accept both for
      one release), so the device and backend can be updated without breaking anything.

## Done

All of these were done on 2026-11-19 and are not committed yet. Tests: `tests/test_evidence_packets.py`
(new, 10), `tests/test_telemetry_packets.py` (new, 7), `tests/test_stream_on_demand.py` (new, 8) and
`tests/test_storage_architecture.py` (5). The full test command is at the end of this section.

- **8. Telemetry `uptime` is now the real uptime.** The packet field is `uptime_ms` (time since the
  process started, which resets on restart) instead of the epoch time.
- **9. Telemetry `hdop`.** It is copied from `gnss_sensor.read()` into the GNSS block when there is a
  fix, and is `null` otherwise. The GNSS sensor module was not changed.
- **10. The backup heartbeat now matches the main packet.** The hand-built dict was replaced by
  `telemetry.build_fallback_full_packet()`, which uses only the last known `latest_sensor` values and
  goes through the same `build_slim_packet()`. It now has the same keys and types as the main packet:
  - `fix` is `"3D_FIX"` / `"NO_FIX"`, and `pwr_src`, `low_batt` and the real `clk_src` are included.
  - `ts` uses the device time zone, like the main packet.
  - Unknown values are `null` instead of `0`.
  - `uptime` is the real uptime.

  The double sequence increment is also fixed: the sequence steps by 1, under the lock.
  - Remaining difference: `hdop` is always `null` in the backup packet, because the last HDOP isn't
    cached in `latest_sensor`.
- **11. On-demand live camera stream, switchable in config.** Done 2026-11-19.
  - `config.py`: `STREAM_ON_DEMAND` (default `True`), `STREAM_DEMAND_PATH`
    (`/api/camera/{device_id}/stream-status`) and `STREAM_DEMAND_POLL_SEC` (3 s).
  - With `True`, `live_frame_streamer()` polls the status endpoint and sends frames only while it
    answers `{"active": true}`. If the backend can't be reached or answers non-200, the stream stays
    off, and this is logged once. Start and pause events are logged.
  - With `False`, it streams continuously while authenticated, exactly as before, and never polls.
  - `live_frame_streamer(..., on_demand=None)` can override the config. The mock-server self-test in
    `tests/test_camera_location_stream.py` passes `on_demand=False`.
  - Backend spec: `prompts/SITE_LIVE_STREAM_PROMPT.txt`. Tests: `tests/test_stream_on_demand.py` (8).
- **Raw GNSS stream moved to its own host** (2026-11-19). `config.GNSS_STREAM_BASE_URL`
  (default `https://solidwasteapinew.scipl.info.in`, env `SWSTP_GNSS_STREAM_URL`) is used for
  `POST /api/{vehicle_id}/live_location`. Login, telemetry, evidence and camera stay on
  `DEFAULT_BACKEND_URL` (`solidwasteapi.scipl.info.in`). Set it to `""` to follow the main backend
  again.
- **12.1** Removed the unused `active_session_query_url` and `last_session_check` from
  `telemetry_streamer()`.
- **12.2** The `raw_gnss_streamer()` docstring and `--gnss-stream-rate` help text now say 5 Hz.

Test run (from `swstp_edge/`, 30 tests, all pass):
`python -m unittest tests.test_stream_on_demand tests.test_telemetry_packets tests.test_evidence_packets tests.test_storage_architecture`

## Closed (no change)

- **12.3 IST timestamps without a UTC offset.** Closed 2026-11-19; kept as is.
- **12.4 Multipart booleans sent as `"True"` / `"False"` and `None` fields omitted.** Closed
  2026-11-19; kept as is.

- **1. Litter backlog replay uploaded the wrong location.**
  - `litter_engine.py`: the sidecar now has top-level `latitude` / `longitude`, which the live
    `upload_item` also reads. `gnss.*` is unchanged.
  - `uploader.py`: falls back to `gnss.latitude` / `gnss.longitude` for litter sidecars already on
    disk. Backlog items (`_from_backlog`) are never tagged with the vehicle's *current* GPS: no fix at
    capture means no location is sent. Live uploads keep the current-GPS fallback as before.
  - Also fixed a crash after a successful upload with no location: the confirmation log did
    `f"{lat:.8f}"` and `get_nearest_house(None, …)` and threw before the local files were deleted,
    which caused repeated re-uploads.
  - No GNSS sensor, telemetry or streaming code was changed.
- **2. Litter idempotency key.** It is generated once per event in `litter_engine.py`, stored in the
  sidecar, and passed in the live `upload_item`, so live and backlog attempts share it.
- **5. `detectionConfidence` of `0.0` was dropped.** Fixed with an explicit `is None` check.
- **6. `motionConfidence` dropped.** It was removed from the upload form, the motion `item_meta`
  (`main.py`) and the backlog fallback metadata.
- **7. `detectionType` tag.** Motion items now set `"detection_type": "MOTION"` (`main.py` and the
  backlog fallback). The uploader also defaults to `MOTION` for older motion sidecars on disk that
  have no tag. Litter stays `"LITTER"`.
