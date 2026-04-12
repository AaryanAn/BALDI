import asyncio
import base64
import json
import os
import threading
from datetime import datetime
from pathlib import Path

import cv2
from nicegui import app, run, ui

from evaluation.letters import LetterEvaluator
from gestures.gestures import Gestures
from tracking.ir_tracker import IRTracker


srcDir = Path(__file__).resolve().parent.parent
recordingsDir = srcDir / "recordings"
recordingsDir.mkdir(exist_ok=True)
path = str(srcDir / "gestures/hand_landmarker.task")
templates_dir = srcDir / "templates"
templates_dir.mkdir(parents=True, exist_ok=True)
logs_dir = srcDir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "evaluations.jsonl"

# Provide built-in templates so users don't have to create their own
if not list(templates_dir.glob("*/*.npy")):
    try:
        from evaluation.font_templates import build_font_templates
        build_font_templates()
    except Exception:
        pass

cap = cv2.VideoCapture(0)
tracker = Gestures(path)
evaluator = LetterEvaluator(templates_dir)

# Optional second camera (IR reflector); may be absent on dev machines.
ir_cap = None
ir_tracker = None
try:
    ir_cap = IRTracker.open_camera(camera_index=1)
    ir_tracker = IRTracker(camera_index=1)
except RuntimeError:
    ir_cap = None
    ir_tracker = None

latest_frame = None
latest_ir_frame = None
raw_frame = None
is_recording = False
video_writer = None
frame_width = 0
frame_height = 0
SHOW_IMAGE_MODEL = os.getenv("BALDI_SHOW_IMAGE_MODEL", "").strip() in {"1", "true", "True", "yes", "YES"}

# Latest raw frame from the gesture camera (written by a daemon thread so cap.read()
# never blocks the NiceGUI asyncio loop).
_gesture_raw = None
_gesture_lock = threading.Lock()

_ir_raw = None
_ir_lock = threading.Lock()

# Which stroke source Evaluate / Save use: "gesture" | "ir"
active_source = "gesture"


def snapshot_active_paths():
    if ir_tracker is not None and active_source == "ir":
        return ir_tracker.snapshot_paths()
    return tracker.snapshot_paths()


def clear_active_paths():
    if ir_tracker is not None and active_source == "ir":
        ir_tracker.clear_path()
    else:
        tracker.clear_path()


def _gesture_capture_thread():
    global _gesture_raw
    while True:
        ok, frame = cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            with _gesture_lock:
                _gesture_raw = frame


def _ir_capture_thread():
    global _ir_raw
    while True:
        ok, frame = ir_cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            with _ir_lock:
                _ir_raw = frame


def process_frame():
    global raw_frame
    with _gesture_lock:
        frame = _gesture_raw
    if frame is None:
        return None

    frame = frame.copy()
    annotated_frame, fingertip = tracker.detect_index_fingertip(frame)

    # Draw path
    for path in tracker.paths:    
        for i in range(1, len(path)):
            cv2.line(annotated_frame,
                    path[i - 1],
                    path[i],
                    (0, 255, 255),
                    3)

    if active_source == "gesture":
        raw_frame = annotated_frame

    _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    return base64.b64encode(buffer).decode("utf-8")


def process_ir_frame():
    """IR camera: blob-tracked reflector tip and stroke paths (non-blocking read via thread)."""
    global raw_frame
    if ir_tracker is None or ir_cap is None:
        return None
    with _ir_lock:
        frame = _ir_raw
    if frame is None:
        return None
    frame = frame.copy()
    annotated_frame, _ = ir_tracker.detect_ir_tip(frame)
    for path_pts in ir_tracker.paths:
        for i in range(1, len(path_pts)):
            cv2.line(
                annotated_frame,
                path_pts[i - 1],
                path_pts[i],
                (0, 255, 255),
                3,
            )

    if active_source == "ir":
        raw_frame = annotated_frame

    _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode("utf-8")


async def background_capture():
    global latest_frame, latest_ir_frame, video_writer, is_recording

    while True:
        # MediaPipe/OpenCV off the asyncio loop so WebSocket heartbeats are not starved.
        frame = await run.io_bound(process_frame)
        if frame:
            latest_frame = frame

        if ir_tracker is not None:
            ir_frame = await run.io_bound(process_ir_frame)
            if ir_frame:
                latest_ir_frame = ir_frame

        if is_recording and video_writer is not None and raw_frame is not None:
            video_writer.write(raw_frame)

        await asyncio.sleep(0.02)


def toggle_record(record_button):
    global is_recording, video_writer, frame_width, frame_height
    if is_recording:
        is_recording = False
        if video_writer is not None:
            video_writer.release()
        video_writer = None
        record_button.props("color=primary")
    else:
        recording_file = str(recordingsDir / f"recording_{len(list(recordingsDir.glob('*.mp4')))}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(recording_file, fourcc, 20.0, (frame_width, frame_height))
        is_recording = True
        record_button.props("color=negative")


@app.on_startup
async def startup():
    global frame_width, frame_height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    threading.Thread(target=_gesture_capture_thread, daemon=True).start()
    if ir_cap is not None:
        threading.Thread(target=_ir_capture_thread, daemon=True).start()
    asyncio.create_task(background_capture())


@app.on_shutdown
def shutdown():
    global video_writer, is_recording
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    is_recording = False
    cap.release()
    if ir_cap is not None:
        ir_cap.release()


@ui.page("/")
def main_page():
    with ui.header().classes("bg-primary items-center justify-between"):
        ui.label("Welcome to BALDI Handwriting").classes("text-h5 font-bold")

        with ui.tabs().classes("absolute-center") as tabs:
            record_tab = ui.tab("New Recording")
            previous_tab = ui.tab("Previous Recordings")

    with ui.tab_panels(tabs, value=record_tab).classes("w-full"):
        with ui.tab_panel(record_tab):
            with ui.row().classes("items-start w-full gap-4 flex-wrap justify-center"):
                with ui.card().style("max-width: 900px; width: 100%;"):
                    source_toggle = None
                    if ir_tracker is not None:
                        source_toggle = ui.toggle(
                            {"gesture": "Hand camera (RGB)", "ir": "IR reflector"},
                            value="gesture",
                        ).classes("q-mb-sm w-full justify-center")

                    gesture_display = ui.interactive_image().style(
                        "width:100%; height:auto; max-height:75vh; object-fit:contain;"
                    )
                    ir_display = None
                    if ir_tracker is not None:
                        ir_display = ui.interactive_image().style(
                            "width:100%; height:auto; max-height:75vh; object-fit:contain; background:#111;"
                        )

                    def sync_source_visibility():
                        global active_source
                        if source_toggle is not None and ir_display is not None:
                            active_source = source_toggle.value
                            gesture_display.set_visibility(active_source == "gesture")
                            ir_display.set_visibility(active_source == "ir")
                        else:
                            active_source = "gesture"

                    sync_source_visibility()
                    if source_toggle is not None:
                        source_toggle.on("update:model-value", lambda _: sync_source_visibility())

                    def update_video():
                        if latest_frame:
                            gesture_display.set_source(
                                f"data:image/jpeg;base64,{latest_frame}"
                            )
                        if ir_display is not None and latest_ir_frame:
                            ir_display.set_source(
                                f"data:image/jpeg;base64,{latest_ir_frame}"
                            )

                    ui.timer(0.03, update_video)

                    with ui.row().classes("w-full justify-center q-mt-sm"):
                        record_button = ui.button("Record")
                        record_button.on_click(lambda: toggle_record(record_button))

                with ui.card().style("min-width: 260px; max-width: 420px; width: 100%;"):
                    ui.label("Built-in templates for A–Z, a–z. You can save your own for better matching.").classes(
                        "text-sm text-grey-7"
                    )
                    ui.label(
                        "Hand camera: keep your other fingers in a loose fist—only thumb and index move. "
                        "Pinch those two fingertips together to record a stroke; open the pinch to finish. "
                        "Dot is red while recording, green when idle."
                    ).classes("text-sm text-grey-7")
                    if ir_tracker is not None:
                        ui.label(
                            "IR camera: hold the reflector tip still briefly to start or stop drawing a stroke "
                            "(green = idle, red = drawing). Choose IR above to use these strokes for Evaluate / Save."
                        ).classes("text-sm text-grey-7")

                    ui.label("Please select your language:")
                    ui.toggle(["English", "Arabic"], value="English")

                    label_input = ui.input("Letter label (optional)").props("clearable")
                    ui.label(
                        "Main score = template match: how similar your stroke is to saved examples of a letter (0–100%). "
                        "Technical confidence among letters is still saved in the evaluation log for debugging."
                    ).classes("text-caption text-grey-6")
                    predicted_label = ui.label("").classes("text-body1 whitespace-pre-line")
                    topk_label = ui.label("").classes("text-sm text-grey-7 whitespace-pre-line")
                    score_label = ui.label("").classes("text-sm text-grey-7 whitespace-pre-line")
                    image_pred_label = ui.label("").classes("text-sm text-grey-7")
                    if not SHOW_IMAGE_MODEL:
                        image_pred_label.set_visibility(False)

                    async def save_template():
                        label = label_input.value or ""
                        paths = snapshot_active_paths()

                        def work():
                            traj = evaluator.get_trajectory(paths)
                            if traj.shape[0] == 0:
                                return {"ok": False, "reason": "no_path"}
                            saved = evaluator.save_template(label, traj)
                            return {"ok": True, "saved": saved}

                        data = await run.io_bound(work)
                        if not data["ok"]:
                            if data.get("reason") == "no_path":
                                ui.notify("No path to save")
                            return
                        saved = data["saved"]
                        if saved is None:
                            ui.notify("Label is empty")
                            return

                        ui.notify(f"Saved template for '{label}'")

                    eval_progress = (
                        ui.linear_progress(show_value=False, size="8px")
                        .classes("w-full")
                        .props("indeterminate color=primary")
                    )
                    eval_progress.set_visibility(False)

                    async def run_evaluation():
                        eval_btn.disable()
                        eval_progress.set_visibility(True)
                        try:
                            label = label_input.value or ""
                            paths = snapshot_active_paths()

                            def work():
                                traj = evaluator.get_trajectory(paths)
                                if traj.shape[0] == 0:
                                    return {"no_path": True}
                                pred = evaluator.predict(traj, top_k=3)
                                result = None
                                if label.strip():
                                    result = evaluator.evaluate(label, traj)
                                image_pred = None
                                if SHOW_IMAGE_MODEL:
                                    try:
                                        from evaluation.letters_image import predict_from_trajectory

                                        image_pred = predict_from_trajectory(traj)
                                    except Exception:
                                        image_pred = {"available": False}
                                return {
                                    "no_path": False,
                                    "pred": pred,
                                    "result": result,
                                    "image_pred": image_pred,
                                }

                            data = await run.io_bound(work)
                            if data.get("no_path"):
                                ui.notify("No path to evaluate")
                                return

                            pred = data["pred"]
                            result = data["result"]
                            image_pred = data["image_pred"]

                            top = pred.get("top") or []
                            best_guess = top[0]["label"] if top else None
                            best_tpl_pct = (
                                int(round(float(top[0]["score"]) * 100.0)) if top else None
                            )

                            if pred.get("predicted_label") is None:
                                if best_guess is not None and best_tpl_pct is not None:
                                    predicted_label.text = (
                                        f"Closest letter by templates: “{best_guess}” ({best_tpl_pct}% match).\n"
                                        f"Not shown as a firm pick — other letters scored close too."
                                    )
                                else:
                                    predicted_label.text = "Could not compare to templates."
                            else:
                                letter = pred["predicted_label"]
                                tpl_pct = None
                                for t in top:
                                    if t.get("label") == letter:
                                        tpl_pct = int(round(float(t["score"]) * 100.0))
                                        break
                                if tpl_pct is None and top:
                                    tpl_pct = int(round(float(top[0]["score"]) * 100.0))
                                tpl_txt = f"{tpl_pct}%" if tpl_pct is not None else "—"
                                predicted_label.text = (
                                    f"Best guess: “{letter}” ({tpl_txt} template match)\n"
                                    f"How well your stroke fits this letter’s saved examples."
                                )

                            try:
                                if top:
                                    lines = []
                                    for i, t in enumerate(top[:3], start=1):
                                        pct = float(t["score"]) * 100.0
                                        lines.append(f"{i}. “{t['label']}”: {pct:.0f}% template match")
                                    topk_label.text = (
                                        "How well your stroke fits each letter’s saved examples:\n"
                                        + "\n".join(lines)
                                    )
                                else:
                                    topk_label.text = ""
                            except Exception:
                                topk_label.text = ""

                            if SHOW_IMAGE_MODEL:
                                if image_pred and image_pred.get("available"):
                                    img_letter = image_pred["predicted_label"]
                                    img_conf = image_pred.get("confidence")
                                    image_pred_label.text = f"Image model: {img_letter} ({img_conf:.2f})"
                                else:
                                    image_pred_label.text = "Image model: not loaded (train with .venv-train)"

                            try:
                                record = {
                                    "ts": datetime.utcnow().isoformat(),
                                    "label": label,
                                    "score": result.get("score") if result else None,
                                    "distance": result.get("distance") if result else None,
                                    "has_templates": result.get("has_templates") if result else None,
                                    "num_templates": result.get("num_templates") if result else None,
                                    "predicted": pred.get("predicted_label"),
                                    "pred_conf": pred.get("confidence"),
                                    "pred_best_dist": pred.get("best_distance"),
                                    "pred_top": pred.get("top"),
                                    "image_pred": image_pred.get("predicted_label") if image_pred else None,
                                    "image_conf": image_pred.get("confidence") if image_pred else None,
                                }
                                with open(log_file, "a") as f:
                                    f.write(json.dumps(record) + "\n")
                            except Exception:
                                pass

                            if not label.strip():
                                score_label.text = ""
                                ui.notify("Prediction done")
                                return

                            if result and not result["has_templates"]:
                                score_label.text = (
                                    f"No template for '{label}'. Use a letter in A–Z / a–z, or save your drawing as a template."
                                )
                                ui.notify("No template for this letter")
                                return

                            if result:
                                score = result["score"]
                                n = result["num_templates"]
                                score_label.text = (
                                    f"Your label “{label}”: {score * 100:.0f}% template match "
                                    f"({n} saved example{'s' if n != 1 else ''}).\n"
                                    f"This only measures fit to “{label}” — not how sure the app is among all letters."
                                )
                            ui.notify("Evaluation done")
                        finally:
                            eval_progress.set_visibility(False)
                            eval_btn.enable()

                    def clear_drawing():
                        clear_active_paths()
                        ui.notify("Path cleared")

                    ui.button("Save as template", on_click=save_template)
                    eval_btn = ui.button("Evaluate", on_click=run_evaluation)
                    ui.button("Clear Drawing", on_click=clear_drawing)

        with ui.tab_panel(previous_tab):
            ui.label("Evaluation log").classes("text-h6")

            @ui.refreshable
            def records_view():
                if not log_file.exists():
                    ui.label("No evaluations yet.")
                    return

                try:
                    lines = log_file.read_text().splitlines()
                except Exception:
                    ui.label("Could not read evaluations log.")
                    return

                if not lines:
                    ui.label("No evaluations yet.")
                    return

                # show the most recent records first
                for line in reversed(lines[-50:]):
                    try:
                        rec = json.loads(line)
                        ts = rec.get("ts", "")
                        lbl = rec.get("label", "")
                        predicted = rec.get("predicted")
                        pred_conf = rec.get("pred_conf")
                        pred_top = rec.get("pred_top")
                        sc = rec.get("score", None)
                        dist = rec.get("distance", None)
                        img_pred = rec.get("image_pred")
                        img_conf = rec.get("image_conf")
                        extra = f"  img={img_pred} ({img_conf})" if img_pred is not None else ""
                        pred_extra = (
                            f"  predicted={predicted} ({pred_conf:.2f})" if predicted is not None and pred_conf is not None else ""
                        )
                        top_extra = ""
                        if isinstance(pred_top, list) and pred_top:
                            try:
                                top_extra = "  top=" + ",".join(
                                    [f"{t.get('label')}:{float(t.get('score') or 0.0):.2f}" for t in pred_top[:3]]
                                )
                            except Exception:
                                top_extra = ""
                        ui.label(f"{ts}  label={lbl}  score={sc}  dist={dist}{pred_extra}{top_extra}{extra}").classes("text-sm")
                    except Exception:
                        continue

            ui.button("Refresh log", on_click=records_view.refresh)
            records_view()

            ui.separator().classes("q-my-md")
            ui.label("Camera recordings (.mp4)").classes("text-h6")

            @ui.refreshable
            def camera_recordings_view():
                if not recordingsDir.exists():
                    ui.label("No recordings folder.")
                    return
                files = sorted(recordingsDir.glob("*.mp4"), key=lambda p: p.name)
                if not files:
                    ui.label("No .mp4 files yet. Use Record on the New Recording tab.")
                    return
                for f in files:
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f.name).classes("text-sm")
                        ui.button(icon="download", on_click=lambda ff=f: ui.download(ff))

            ui.button("Refresh list", on_click=camera_recordings_view.refresh)
            camera_recordings_view()
