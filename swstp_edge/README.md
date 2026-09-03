# SWSTP Edge — Raspberry Pi 4B Firmware

Single-device replacement for the original Arduino Uno + PC two-device stack.
The DS3231 RTC, MPU-6500 IMU, and NEO-6M GNSS module are wired directly to
the Pi's I²C bus and hardware UART.  All downstream behaviour (motion detection,
geofencing, backend API contract) is unchanged.

---

## Directory Structure

```
swstp_edge/
├── main.py                  ← Entry point  (python3 main.py)
├── config.py                ← All constants, GPIO pin map, device ID loader
├── device_config.json       ← Hardware identity (replaces Arduino EEPROM)
├── roi_polygon.json         ← Persistent ROI polygon (same format as before)
├── requirements.txt
├── captures/                ← Motion evidence frames saved locally
├── sensors/
│   ├── rtc.py               ← DS3231 via kernel i2c-rtc overlay
│   ├── rtc_sync.py          ← Standalone NTP/RTC initial sync & periodic backup
│   ├── imu.py               ← MPU-6500 via smbus2 + ported complementary filter
│   ├── gnss.py              ← NEO-6M via gpsd + gpsdclient
│   ├── leds.py              ← Status LEDs via gpiozero
│   └── power.py             ← Secondary battery supervisor, evidence flusher & safe shutdown
├── telemetry.py             ← 20 Hz sensor → JSON packet builder (slim & full)
├── motion.py                ← Motion detection (unchanged algorithm)
├── geofence.py              ← Haversine, road-snap, safe-zone (unchanged)
└── network/
    ├── uploader.py          ← Backend HTTP workers (unchanged endpoints)
    └── location_fallback.py ← IP-geolocation fallback (Linux-only)
```

---

## Hardware Wiring

### DS3231 RTC — I²C (address 0x68)

| DS3231 pin | Pi 40-pin header |
|-----------|-----------------|
| VCC       | Pin 1 (3.3 V)   |
| GND       | Pin 6 (GND)     |
| SDA       | Pin 3 (GPIO 2)  |
| SCL       | Pin 5 (GPIO 3)  |

> ⚠️ **VOLTAGE CAUTION**: Pi I²C operates at **3.3 V only**.  Many cheap DS3231
> breakout boards tie their pull-up resistors to VCC = 5 V, which will pull the
> SDA/SCL lines to 5 V — **exceeding the Pi GPIO absolute maximum of 3.3 V**.
> Use a bidirectional 3.3 V ↔ 5 V I²C level-shifter (e.g. TXB0102, BSS138-
> based module) if your breakout uses 5 V pull-ups.  Breakout boards whose
> pull-ups are tied to 3.3 V (e.g. Adafruit #3013) are safe for direct
> connection.

---

### MPU-6500 IMU — I²C (address 0x69, AD0 pulled HIGH)

| MPU-6500 pin | Pi 40-pin header |
|-------------|-----------------|
| VCC         | Pin 1 (3.3 V)   |
| GND         | Pin 6 (GND)     |
| SDA         | Pin 3 (GPIO 2)  |
| SCL         | Pin 5 (GPIO 3)  |
| AD0         | 3.3 V (HIGH)    |

> ⚠️ **VOLTAGE CAUTION**: Most GY-6500 / GY-521 breakout boards include an
> on-board 3.3 V regulator and run I²C logic at 3.3 V — **safe for direct Pi
> connection when VCC is wired to 3.3 V**.  If you power the module from 5 V,
> the I²C logic levels will be 5 V — insert a level-shifter before connecting
> to the Pi.  Always verify your specific module's schematic.

---

### NEO-6M GNSS — Hardware UART (/dev/serial0)

| NEO-6M pin | Pi 40-pin header |
|-----------|-----------------|
| VCC       | Pin 1 (3.3 V) or Pin 4 (5 V) — check module |
| GND       | Pin 6 (GND)     |
| TX        | Pin 10 (GPIO 15 / UART0 RX) |
| RX        | Pin 8  (GPIO 14 / UART0 TX) |

> ⚠️ **VOLTAGE CAUTION**:
> - **NEO-6M TX → Pi RX**: The NEO-6M output is typically 2.8 V logic — safe
>   for direct connection to the Pi 3.3 V GPIO.
> - **Pi TX → NEO-6M RX**: The Pi TX output is 3.3 V.  Most NEO-6M modules
>   accept 3.3 V on their RX input.  However, some inexpensive modules labelled
>   "5 V" may have the RX pin connected to 5 V-level circuitry.  **Check your
>   module's schematic.**  If uncertain, use a 1 kΩ / 2 kΩ voltage divider on
>   the Pi TX line (Pi TX → 1 kΩ → NEO-6M RX; NEO-6M RX → 2 kΩ → GND), or a
>   logic-level shifter.

---

### Status LEDs (BCM GPIO, configurable in config.py)

| LED      | Default BCM GPIO | Behaviour                              |
|----------|-----------------|----------------------------------------|
| RTC_GREEN  | 17            | Solid ON = RTC OK; blink = RTC fault  |
| IMU_GREEN  | 27            | Solid ON = IMU OK; blink = IMU fault  |
| GNSS_GREEN | 22            | Solid ON = GNSS fix; blink = no fix   |
| YELLOW     | 23            | 1 Hz blink when all modules OK        |
| RED        | 24            | Blinks when any module is faulting    |

Wire each LED through a **220 Ω – 470 Ω current-limiting resistor** to GND.
Pi GPIO outputs are 3.3 V at up to ~16 mA per pin.

---

### Power Management & Secondary Battery Deployment

In field deployment, the Pi is powered by the vehicle's main power supply (12V/24V via DC-DC converter). When the vehicle is turned off, a secondary battery (UPS HAT, Supercapacitor, or 18650 Li-ion pack) keeps the Pi running temporarily.

#### 1. Low-Battery Detection & Emergency Evidence Flush
- **Alert Pin**: Connect the UPS or battery BMS "Low Battery" / "Power Fail" output to **GPIO 25 (Pin 22)** (configurable in `config.py`).
- **Logic**: Active-LOW by default (pulled to GND on low battery warning).
- When low battery is detected:
  1. **Camera feed halted**: Prevents new capture items.
  2. **Evidence Flushed**: The worker immediately uploads all pending images in `upload_queue` to `POST /api/evidence/upload`.
  3. **Telemetry Flushed**: Telemetry queue is drained and a final status is posted.
  4. **Filesystem Synced**: `os.sync()` is executed to ensure SD card integrity.
  5. **Safe OS Shutdown**: Calls `sudo shutdown -h now` so the operating system safely halts.

#### 2. Auto-Boot When Vehicle Powers On Again
When the vehicle is restarted, the Pi must boot up automatically without manual intervention. There are two standard hardware methods:

- **Method A (Power Cycle — Recommended with Automotive UPS / Relay)**:
  Once the Pi halts (`shutdown -h now`), an automotive UPS module or timer relay completely cuts the 5V rail after a short delay (e.g., 30s). When the vehicle turns on again, 12V ignition power returns, 5V power is re-applied from 0V, and the Raspberry Pi **automatically boots by hardware design**.
- **Method B (GPIO 3 / RUN Wake-up)**:
  If 5V remains energized to the halted Pi, momentarily pulling **GPIO 3 (Pin 5)** or the **GLOBAL_EN / RUN** header pin to GND triggers an immediate hardware boot. An optocoupler or small NPN transistor connected to the vehicle's ignition signal (ACC) can pulse GPIO 3 or RUN when the key turns.

---


## One-time Pi OS Setup

### 1. Enable I²C and hardware UART

Edit `/boot/config.txt` (or `/boot/firmware/config.txt` on newer Pi OS):

```ini
# Enable I2C
dtparam=i2c_arm=on

# Discipline system clock from DS3231 via kernel driver (no app-level byte-banging)
dtoverlay=i2c-rtc,ds3231

# Enable hardware UART on GPIO 14/15 for NEO-6M
enable_uart=1
# Free /dev/serial0 from the Bluetooth chip (Pi 3/4 only)
dtoverlay=disable-bt
```

Reboot after editing.

### 2. Sync hardware clock on boot

```bash
sudo apt install i2c-tools
sudo hwclock --hctosys      # load RTC → system clock (done automatically at boot by Raspberry Pi OS)
date                        # verify system time is correct
i2cdetect -y 1             # should show 0x68 (DS3231) and 0x69 (MPU-6500)
```

### 3. Install and configure gpsd

```bash
sudo apt install gpsd gpsd-clients python3-gps
```

Edit `/etc/default/gpsd`:

```bash
DEVICES="/dev/serial0"
GPSD_OPTIONS="-n"
START_DAEMON="true"
USBAUTO="false"
```

```bash
sudo systemctl enable gpsd
sudo systemctl restart gpsd
cgps -s         # verify satellite data flowing
```

### 4. Install Python dependencies

```bash
cd swstp_edge/
pip3 install -r requirements.txt
```

---

## Device Identity Provisioning

Edit `device_config.json` to set the device code:

```json
{
  "device_id": "SWSTP-PI-001"
}
```

This replaces the EEPROM `PROVISION:` command from the Arduino firmware.

---

## Running

```bash
cd swstp_edge/
python3 main.py
```

### CLI flags (all original flags preserved)

| Flag | Default | Description |
|------|---------|-------------|
| `--headless` | off | Run without cv2.imshow window |
| `--source N` | 0 | Camera index or MP4 path |
| `--device-id ID` | from `device_config.json` | Override device code |
| `--backend-url URL` | auto-detect | SWSTP backend URL |
| `--ulb-id ID` | `ULB_MH_AMRAVATI` | ULB identifier |
| `--session-id N` | 0 (auto) | Operational session ID |
| `--fps-stream F` | 30 | Live frame stream FPS to backend |
| `--save-dir DIR` | `./captures` | Local capture directory |
| `--no-gps-fallback` | off | Disable IP-geolocation fallback |
| `--gps-fallback-timeout S` | 20 | Seconds before fallback engages |
| `--no-power-monitor` | off | Disable background battery monitor thread |
| `--simulate-low-batt-sec N` | 0 | Simulate low-battery shutdown after N seconds (testing) |
| `--port` / `--baud` | ignored | Legacy flags — no-op on Pi |


### Interactive controls (GUI mode)

| Key | Action |
|-----|--------|
| `r` | Toggle ROI polygon plotting mode / connect & save |
| `t` | Toggle camera feed on/off |
| `c` | Clear in-progress polygon points |
| `q` | Quit |

Mouse: **left-click** to place polygon vertices; **right-click** or click near P1 to close & save.

---

## Backend API Contract

The packet shapes sent to all three backend endpoints are **byte-for-byte
identical** to what the original `webcam_motion_detect.py` produced:

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/api/camera/{deviceId}/frame` | POST | JPEG, `Content-Type: image/jpeg` |
| `/api/telemetry/ingest-batch`  | POST | JSON batch, same field names/units |
| `/api/evidence/upload`         | POST | multipart/form-data, same metadata fields |

---

## Architecture Comparison

| Component | Old (Arduino + PC) | New (Raspberry Pi 4B) |
|-----------|-------------------|-----------------------|
| Device identity | Arduino EEPROM | `device_config.json` |
| RTC | DS3231 → Arduino → USB serial | DS3231 → kernel `i2c-rtc` → `datetime.now()` |
| IMU | MPU-6500 → Arduino → USB serial | MPU-6500 → `smbus2` registers + ported complementary filter |
| GNSS | NEO-6M → Arduino SoftwareSerial → USB serial | NEO-6M → `/dev/serial0` → `gpsd` → `gpsdclient` |
| Telemetry → PC | 115200 baud JSON lines | Native Python thread, 20 Hz |
| Motion detection | `webcam_motion_detect.py` on PC | `motion.py` on Pi (unchanged algorithm) |
| LED control | Arduino `digitalWrite` | `gpiozero` |
| Location fallback | Windows GPS + IP-geolocation | IP-geolocation only (Linux) |
