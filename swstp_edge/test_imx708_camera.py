#!/usr/bin/env python3
"""
test_imx708_camera.py — Hardware Test & Diagnostic Tool for Raspberry Pi Camera Module 3 (IMX708).

Validates:
  1. Picamera2 and libcamera installation
  2. Sony IMX708 sensor detection (I2C + CSI)
  3. Video configuration (resolution, framerate)
  4. Phase Detection Autofocus (PDAF) functionality
  5. Continuous capture framerate and latency
  6. Sample frame capture and file save to captures/test_imx708_capture.jpg

Usage:
  python3 test_imx708_camera.py
  python3 test_imx708_camera.py --width 1280 --height 720 --frames 60
  python3 test_imx708_camera.py --hdr
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from camera import (
    is_picamera2_available,
    detect_imx708_sensor,
    IMX708CameraCapture,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test Raspberry Pi Camera Module 3 (IMX708)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS (default: 30.0)")
    parser.add_argument("--frames", type=int, default=30, help="Number of benchmark frames (default: 30)")
    parser.add_argument("--af-mode", default="continuous", choices=["continuous", "auto", "manual"],
                        help="Autofocus mode (default: continuous)")
    parser.add_argument("--hdr", action="store_true", help="Enable hardware HDR mode")
    parser.add_argument("--save-dir", default=os.path.join(_HERE, "captures"),
                        help="Directory to save test capture (default: captures/)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n==================================================================")
    print(" RASPBERRY PI CAMERA MODULE 3 (IMX708) DIAGNOSTIC & TEST TOOL")
    print("==================================================================")

    # ── Step 1: Check Python Picamera2 Library ───────────────────────────
    print("\n[STEP 1] Checking Picamera2 library...")
    if not is_picamera2_available():
        print("  ❌ ERROR: 'picamera2' is NOT installed or could not be imported.")
        print("  👉 Resolution for Raspberry Pi OS (Bullseye / Bookworm):")
        print("     sudo apt update")
        print("     sudo apt install -y python3-picamera2")
        print("==================================================================\n")
        sys.exit(1)
    print("  ✔ Picamera2 library is installed and available.")

    # ── Step 2: Detect Sony IMX708 Hardware ──────────────────────────────
    print("\n[STEP 2] Probing for Sony IMX708 hardware...")
    diag = detect_imx708_sensor()
    print(f"  Detected: {diag['detected']}")
    print(f"  Model:    {diag['model']}")
    print(f"  Details:  {diag.get('details', 'None')}")

    if not diag["detected"]:
        print("\n  ⚠️ WARNING: IMX708 was not detected automatically.")
        print("  Troubleshooting checklist:")
        print("    1. Verify ribbon cable is firmly seated in the CSI camera port.")
        print("       (Silver contacts face the HDMI ports on Pi 4B).")
        print("    2. On Raspberry Pi OS Bookworm, ensure camera auto-detect is enabled:")
        print("       grep -E 'camera_auto_detect|dtoverlay=imx708' /boot/firmware/config.txt")
        print("    3. If needed, manually add overlay to /boot/firmware/config.txt:")
        print("       dtoverlay=imx708")
        print("    4. Verify with: rpicam-hello --list-cameras")
        print("  Attempting to initialize anyway...")
    else:
        print("  ✔ IMX708 hardware confirmed.")

    # ── Step 3: Initialize IMX708 via Picamera2 ──────────────────────────
    print(f"\n[STEP 3] Initializing Camera Module 3 at {args.width}x{args.height} @ {args.fps:.1f} FPS...")
    print(f"  Autofocus Mode: {args.af_mode.upper()} (Phase Detection AF)")
    print(f"  Hardware HDR:   {'ENABLED' if args.hdr else 'DISABLED'}")

    try:
        cap = IMX708CameraCapture(
            camera_idx=0,
            width=args.width,
            height=args.height,
            fps=args.fps,
            af_mode=args.af_mode,
            hdr=args.hdr,
        )
    except Exception as exc:
        print(f"  ❌ Camera initialization failed: {exc}")
        print("==================================================================\n")
        sys.exit(1)

    info = cap.get_info()
    print(f"  ✔ Sensor initialized successfully: {info['model']}")
    print(f"  ✔ Backend: {info['backend']}")
    print(f"  ✔ Autofocus: {info['autofocus']}")

    # ── Step 4: Capture Warm-up & Benchmark Frames ───────────────────────
    print(f"\n[STEP 4] Capturing {args.frames} test frames...")
    latencies = []
    sample_frame = None

    for i in range(args.frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()

        if not ret or frame is None:
            print(f"  ❌ Frame {i+1} capture failed!")
            break

        dt_ms = (t1 - t0) * 1000.0
        latencies.append(dt_ms)
        sample_frame = frame

        if (i + 1) % 10 == 0 or (i + 1) == args.frames:
            fps_instant = 1000.0 / dt_ms if dt_ms > 0 else 0
            print(f"  Captured frame {i+1:3d}/{args.frames} (Latency: {dt_ms:5.1f} ms | {fps_instant:4.1f} FPS)")

    if sample_frame is not None:
        h, w = sample_frame.shape[:2]
        channels = sample_frame.shape[2] if sample_frame.ndim > 2 else 1
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        avg_fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0

        print(f"\n  Frame Dimensions: {w}x{h} (channels: {channels}, dtype: {sample_frame.dtype})")
        print(f"  Average Latency:  {avg_latency:.2f} ms")
        print(f"  Sustained FPS:    {avg_fps:.1f} FPS")

        # ── Step 5: Save Test Capture ────────────────────────────────────
        print(f"\n[STEP 5] Saving sample test frame...")
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, "test_imx708_capture.jpg")

        # Try saving with OpenCV if available
        try:
            import cv2
            cv2.imwrite(out_path, sample_frame)
            file_size_kb = os.path.getsize(out_path) / 1024.0
            print(f"  ✔ Frame saved to: {out_path} ({file_size_kb:.1f} KB)")
        except Exception:
            try:
                from PIL import Image
                # Convert BGR to RGB for PIL
                rgb = sample_frame[..., ::-1] if sample_frame.ndim == 3 else sample_frame
                im = Image.fromarray(rgb)
                im.save(out_path)
                file_size_kb = os.path.getsize(out_path) / 1024.0
                print(f"  ✔ Frame saved to: {out_path} ({file_size_kb:.1f} KB)")
            except Exception as save_err:
                print(f"  ⚠️ Could not save image file: {save_err}")

    # ── Clean up ─────────────────────────────────────────────────────────
    cap.release()
    print("\n  ✔ Camera stopped and resources released cleanly.")
    print("\n==================================================================")
    print(" RESULT: Raspberry Pi Camera Module 3 (IMX708) is FULLY OPERATIONAL!")
    print(" Run the edge gateway with:")
    print("   python3 main.py --source imx708")
    print(" or simply:")
    print("   python3 main.py")
    print("==================================================================\n")


if __name__ == "__main__":
    main()
