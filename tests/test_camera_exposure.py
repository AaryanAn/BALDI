"""
Lock camera exposure on macOS via AVFoundation (PyObjC), then attempt to
lower shutter speed and ISO/gain.

Setup:
    pip install pyobjc-framework-AVFoundation pyobjc-framework-CoreMedia
    python tests/test_camera_exposure.py [camera_index]

Press Q to quit the live preview.
"""

import sys
import time
import cv2

CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def configure_camera(camera_index: int) -> None:
    import AVFoundation as AVF
    import CoreMedia    as CM

    devices = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)
    print(f"\nAVFoundation devices:")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d.localizedName()}")

    device = devices[camera_index]

    # ── Report what this device exposes ──────────────────────────────────────
    print(f"\nDevice: {device.localizedName()}")
    print(f"  ISO current          : {device.ISO()}")
    print(f"  exposureDuration     : {CM.CMTimeGetSeconds(device.exposureDuration()):.6f} s")
    print(f"  exposureTargetBias   : {device.exposureTargetBias()}")
    print(f"  exposureTargetBias   range: {device.minExposureTargetBias()} – {device.maxExposureTargetBias()}")
    print(f"  activeFormat         : {device.activeFormat()}")

    formats = device.formats()
    print(f"\n  Supported frame rate ranges (all formats):")
    seen = set()
    for fmt in formats:
        for r in fmt.videoSupportedFrameRateRanges():
            lo  = r.minFrameRate()
            hi  = r.maxFrameRate()
            key = (round(lo, 1), round(hi, 1))
            if key not in seen:
                seen.add(key)
                print(f"    {lo:.1f} – {hi:.1f} fps")

    success, err = device.lockForConfiguration_(None)
    if not success:
        print(f"\nERROR: lockForConfiguration_ failed: {err}")
        return

    # ── 1. Lock exposure mode ─────────────────────────────────────────────────
    if device.isExposureModeSupported_(AVF.AVCaptureExposureModeLocked):
        device.setExposureMode_(AVF.AVCaptureExposureModeLocked)
        print("\n[OK] AVCaptureExposureModeLocked applied")
    else:
        print("\n[!!] Locked mode not supported")

    # ── 2. Try exposure target bias (was 0–0 last time, try anyway) ───────────
    min_bias = device.minExposureTargetBias()
    max_bias = device.maxExposureTargetBias()
    if min_bias < 0:
        device.setExposureTargetBias_completionHandler_(min_bias, None)
        print(f"[OK] exposureTargetBias set to {min_bias:.2f} EV")
    else:
        print(f"[--] exposureTargetBias not usable (range {min_bias}–{max_bias})")

    # ── 3. Try forcing a high frame rate (constrains max shutter time) ────────
    # CMTimeMake(1, fps) — higher fps = shorter max exposure
    TARGET_FPS = 60
    min_dur = CM.CMTimeMake(1, TARGET_FPS)
    try:
        device.setActiveVideoMinFrameDuration_(min_dur)
        device.setActiveVideoMaxFrameDuration_(min_dur)
        actual = CM.CMTimeGetSeconds(device.activeVideoMinFrameDuration())
        print(f"[OK] Frame duration set to 1/{TARGET_FPS} s "
              f"(actual: {actual:.4f} s = {1/actual:.1f} fps) "
              f"— forces shutter ≤ {1000/TARGET_FPS:.1f} ms")
    except Exception as e:
        print(f"[--] setActiveVideoMinFrameDuration_ failed: {e}")

    # ── 4. Try setting ISO via custom exposure (Custom mode may be refused) ───
    if device.isExposureModeSupported_(AVF.AVCaptureExposureModeCustom):
        min_iso = device.activeFormat().minISO()
        cur_dur = device.exposureDuration()
        device.setExposureModeCustomWithDuration_ISO_completionHandler_(
            cur_dur, min_iso, None
        )
        print(f"[OK] Custom mode: ISO set to minimum ({min_iso})")
    else:
        print(f"[--] Custom exposure mode not supported — cannot set ISO directly")

    device.unlockForConfiguration()


def live_preview(cap: cv2.VideoCapture) -> None:
    print("\nLive feed — press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        mean = frame.mean()
        cv2.putText(frame, f"mean brightness: {mean:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Exposure test (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    print(f"Opening camera {CAMERA_INDEX} ...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: could not open camera.")
        sys.exit(1)
    print(f"Backend: {cap.getBackendName()}")

    configure_camera(CAMERA_INDEX)
    time.sleep(0.5)
    live_preview(cap)
    cap.release()


if __name__ == "__main__":
    main()
