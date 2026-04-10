import asyncio
import base64
import threading
import cv2
import os

from nicegui import ui, app
from gestures.gestures import Gestures
from tracking.ir_tracker import IRTracker

from pathlib import Path


recordingsDir = Path(__file__).resolve().parent.parent / "recordings"
recordingsDir.mkdir(exist_ok=True)
srcDir = Path(__file__).resolve().parent.parent
path = str(srcDir / "gestures/hand_landmarker.task")

# ── Gesture tracker (integrated webcam, index 0) ──────────────────────────
cap     = cv2.VideoCapture(1)
tracker = Gestures(path)

# ── IR tracker (ELP IR camera, index 1) ──────────────────────────────────
ir_tracker = IRTracker(camera_index=0)
ir_cap     = IRTracker.open_camera(camera_index=0)

width  = 0
height = 0

# ── Shared state written by camera threads, read by async loop ───────────
# Each camera runs in its own daemon thread so cap.read() blocking never
# stalls the NiceGUI asyncio event loop.  Threads write raw frames;
# the async loop reads them, runs detection, and updates the UI globals.
_gesture_raw: cv2.typing.MatLike | None = None
_ir_raw:      cv2.typing.MatLike | None = None
_gesture_lock = threading.Lock()
_ir_lock      = threading.Lock()

raw_frame      = None          # frame written to video recorder
latest_frame   = None          # gesture camera JPEG (base64)
latest_ir_frame = None         # IR camera JPEG (base64)
is_recording   = False
video_writer   = None
image          = None

# Active display source: "gesture" | "ir"
active_source = "gesture"


# ── Camera capture threads ────────────────────────────────────────────────

def _gesture_capture_thread():
    """Spin as fast as the camera hardware allows; store the latest frame."""
    global _gesture_raw
    while True:
        ok, frame = cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            with _gesture_lock:
                _gesture_raw = frame


def _ir_capture_thread():
    """Same as above but for the ELP IR camera."""
    global _ir_raw
    while True:
        ok, frame = ir_cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            with _ir_lock:
                _ir_raw = frame


# ── Processing helpers (called from the async loop) ───────────────────────

def process_frame() -> str | None:
    """
    Grab the latest gesture-camera frame (non-blocking — already captured
    by the background thread), run fingertip detection, and return a
    base64-encoded JPEG string.
    """
    global raw_frame
    with _gesture_lock:
        frame = _gesture_raw
    if frame is None:
        return None

    frame = frame.copy()
    annotated_frame, _ = tracker.detect_index_fingertip(frame)

    # Draw path overlay
    for path_pts in tracker.paths:
        for i in range(1, len(path_pts)):
            cv2.line(annotated_frame,
                     path_pts[i - 1],
                     path_pts[i],
                     (0, 255, 255), 3)

    if active_source == "gesture":
        raw_frame = annotated_frame

    _, buffer = cv2.imencode(
        ".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )
    return base64.b64encode(buffer).decode("utf-8")


def process_ir_frame() -> str | None:
    """
    Grab the latest IR-camera frame (non-blocking), run blob detection
    via IRTracker, and return a base64-encoded JPEG string.

    Overlay:
      - Green ring + crosshair + dot = idle
      - Red  ring + crosshair + dot = writing mode active
      - Cyan path lines for drawn strokes
    """
    global raw_frame
    with _ir_lock:
        frame = _ir_raw
    if frame is None:
        return None

    frame = frame.copy()
    annotated_frame, _ = ir_tracker.detect_ir_tip(frame)

    # Draw path overlay
    for path_pts in ir_tracker.paths:
        for i in range(1, len(path_pts)):
            cv2.line(annotated_frame,
                     path_pts[i - 1],
                     path_pts[i],
                     (0, 255, 255), 3)

    if active_source == "ir":
        raw_frame = annotated_frame

    _, buffer = cv2.imencode(
        ".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )
    return base64.b64encode(buffer).decode("utf-8")


# ── Async processing loop ─────────────────────────────────────────────────

async def background_capture():
    """
    Run detection on the latest camera frames as fast as possible.
    Camera reads happen in dedicated threads so this loop is never
    blocked by hardware I/O.
    """
    global latest_frame, latest_ir_frame, video_writer, is_recording, raw_frame

    while True:
        frame = process_frame()
        if frame:
            latest_frame = frame

        ir_frame = process_ir_frame()
        if ir_frame:
            latest_ir_frame = ir_frame

        if is_recording and video_writer and raw_frame is not None:
            video_writer.write(raw_frame)

        # Very short yield — lets NiceGUI service UI events between frames.
        # Detection itself takes ~2-5 ms so total loop is ~5-8 ms → ~30 fps.
        await asyncio.sleep(0)


# ── App lifecycle ─────────────────────────────────────────────────────────

def toggle_record(record_button):
    global is_recording, video_writer, width, height
    if is_recording:
        is_recording = False
        video_writer.release()
        video_writer = None
        record_button.props("color=primary")
    else:
        recordingFile = str(
            recordingsDir / f"recording_{len(os.listdir(recordingsDir))}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(recordingFile, fourcc, 20.0, (width, height))
        is_recording = True
        record_button.props("color=negative")


@app.on_startup
async def startup():
    global width, height
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Start camera threads (daemon=True so they die with the process)
    threading.Thread(target=_gesture_capture_thread, daemon=True).start()
    threading.Thread(target=_ir_capture_thread,      daemon=True).start()

    asyncio.create_task(background_capture())


@app.on_shutdown
def shutdown():
    cap.release()
    ir_cap.release()


# ── UI ────────────────────────────────────────────────────────────────────

@ui.page("/")
def main_page():
    global active_source

    with ui.header().classes("bg-primary items-center justify_between"):
        ui.label("Welcome to BALDI Handwriting").classes(
            "text-h5 font-bold items-center justify-between"
        )
        with ui.tabs().classes("absolute-center") as tabs:
            recordTab             = ui.tab("New Recording")
            previousRecordingsTab = ui.tab("Previous Recordings")

    with ui.tab_panels(tabs, value=recordTab).classes("w-full"):
        with ui.tab_panel(recordTab):
            with ui.row():
                # ── Source selector ────────────────────────────────────────
                with ui.card().classes("w-full justify-center items-center"):
                    source_toggle = ui.toggle(
                        {"gesture": "Hand Gesture (RGB)", "ir": "IR Reflector"},
                        value="gesture",
                    ).classes("q-mb-md")

                    # Gesture display
                    gesture_display = ui.interactive_image().style(
                        "height:70vh; width: auto; background: #000;"
                    )
                    ui.timer(
                        0.03,
                        lambda: gesture_display.set_source(
                            f"data:image/jpeg;base64,{latest_frame}"
                        ) if latest_frame and source_toggle.value == "gesture" else None,
                    )

                    # IR display
                    ir_display = ui.interactive_image().style(
                        "height:70vh; width: auto; background: #111;"
                    )
                    ui.timer(
                        0.03,
                        lambda: ir_display.set_source(
                            f"data:image/jpeg;base64,{latest_ir_frame}"
                        ) if latest_ir_frame and source_toggle.value == "ir" else None,
                    )

                    def _update_visibility():
                        global active_source
                        active_source = source_toggle.value
                        gesture_display.set_visibility(active_source == "gesture")
                        ir_display.set_visibility(active_source == "ir")

                    source_toggle.on("update:model-value", lambda _: _update_visibility())
                    _update_visibility()

                record_button = ui.button("Record")
                record_button.on_click(lambda: toggle_record(record_button))

                def clear_drawing():
                    if source_toggle.value == "ir":
                        ir_tracker.clear_path()
                    else:
                        tracker.clear_path()
                    ui.notify("Path cleared!")

                ui.button("Clear Drawing", on_click=clear_drawing)

                with ui.card():
                    ui.label("Please select your language:")
                    ui.toggle(["English", "Arabic"], value="English")

        with ui.tab_panel(previousRecordingsTab):
            ui.label("Saved Gestures").classes("text-h6")

            def refresh_list():
                list_container.clear()
                with list_container:
                    files = sorted(recordingsDir.glob("*.mp4"))
                    for f in files:
                        with ui.row().classes("items-center"):
                            ui.label(f.name)
                            ui.button(icon="download", on_click=lambda f=f: ui.download(f))

            list_container = ui.column()
            ui.button("Refresh List", on_click=refresh_list)
            ui.timer(0.1, refresh_list, once=True)