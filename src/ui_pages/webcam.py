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
from ui_pages.recording_paths import next_recording_path
from ui_pages.stroke_source import clear_active_paths as _clear_active_paths
from ui_pages.stroke_source import snapshot_active_paths as _snapshot_active_paths


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

# IR camera only: separate thread so cap.read() on device 1 never blocks the loop.
_ir_raw = None
_ir_lock = threading.Lock()

# Which stroke source Evaluate / Save use: "gesture" | "ir"
active_source = "gesture"


def snapshot_active_paths():
    return _snapshot_active_paths(active_source, tracker, ir_tracker)


def clear_active_paths():
    _clear_active_paths(active_source, tracker, ir_tracker)


def _ir_capture_thread():
    global _ir_raw
    while True:
        ok, frame = ir_cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            with _ir_lock:
                _ir_raw = frame


def process_frame():
    """Same pipeline as ML_testing: read + flip + MediaPipe inside io_bound (full pinch rate)."""
    global raw_frame
    success, frame = cap.read()
    if not success:
        return None

    frame = cv2.flip(frame, 1)

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
        # Hand camera: single io_bound like ML_testing (pinch + tracking stay in sync with frames).
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
        recording_file = str(next_recording_path(recordingsDir))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(recording_file, fourcc, 20.0, (frame_width, frame_height))
        is_recording = True
        record_button.props("color=negative")


@app.on_startup
async def startup():
    global frame_width, frame_height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    try:
        ui.colors(primary="#0d3b66", secondary="#4a6fa5")
    except Exception:
        pass

    ui.add_head_html(
        """
        <style>
          body {
            background: linear-gradient(165deg, #e4edf5 0%, #d8e6f0 38%, #cfdce8 100%) !important;
            min-height: 100vh;
          }
          .q-tab-panels { padding: 12px 16px 24px !important; }
          .baldi-app-card {
            border-radius: 14px;
            border: 1px solid rgba(13, 59, 102, 0.12);
            box-shadow: 0 4px 24px rgba(13, 59, 102, 0.07), 0 1px 3px rgba(13, 59, 102, 0.06);
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
          }
          .baldi-app-card--accent {
            border-left: 4px solid #0d3b66;
          }
          .baldi-video-wrap {
            border-radius: 14px;
            overflow: hidden;
            background: radial-gradient(ellipse at center, #1a2836 0%, #0f1419 70%);
            border: 1px solid rgba(13, 59, 102, 0.35);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04), 0 8px 32px rgba(0,0,0,0.12);
          }
          .baldi-pill {
            letter-spacing: 0.08em;
            font-size: 0.65rem;
            text-transform: uppercase;
            color: #5c7a94;
            font-weight: 600;
          }
          .baldi-header-bar {
            background: linear-gradient(90deg, #0d3b66 0%, #164773 50%, #0d3b66 100%) !important;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 4px 20px rgba(13, 59, 102, 0.25);
          }
          .baldi-distance-tip {
            background: linear-gradient(125deg, rgba(13, 59, 102, 0.07) 0%, rgba(46, 196, 182, 0.08) 100%);
            border: 1px solid rgba(13, 59, 102, 0.14);
            border-radius: 12px;
            padding: 12px 14px;
          }
          .baldi-page-shell {
            max-width: 1280px;
            margin-left: auto;
            margin-right: auto;
          }
        </style>
        """,
        shared=True,
    )

    with ui.header().classes("baldi-header-bar text-white"):
        with ui.row().classes("w-full items-center justify-between q-px-md q-py-md no-wrap baldi-page-shell"):
            with ui.column().classes("gap-none"):
                ui.label("BALDI").classes("text-h6 text-weight-bold").style("letter-spacing: 0.04em;")
                ui.label("Handwriting in the air").classes("text-caption").style(
                    "opacity: 0.88; color: #c5d9ec; line-height: 1.25;"
                )
            with ui.tabs().classes("text-white").props("dense indicator-color=white").style(
                "background: transparent;"
            ) as tabs:
                record_tab = ui.tab("New recording")
                previous_tab = ui.tab("History")

    with ui.tab_panels(tabs, value=record_tab).classes("w-full bg-transparent"):
        with ui.tab_panel(record_tab):
            with ui.row().classes(
                "baldi-page-shell items-start justify-center w-full q-col-gutter-lg flex-wrap"
            ).style("max-width: 100%; margin: 0 auto;"):
                with ui.column().classes("col").style("flex: 1 1 280px; max-width: min(92vw, 960px); min-width: 240px;"):
                    with ui.card().classes("baldi-app-card q-pa-lg w-full"):
                        with ui.row().classes("items-baseline justify-between q-mb-sm no-wrap"):
                            ui.label("Camera preview").classes(
                                "text-subtitle2 text-weight-medium text-grey-9"
                            )
                            ui.label("RGB").classes("baldi-pill")
                        source_toggle = None
                        if ir_tracker is not None:
                            source_toggle = ui.toggle(
                                {"gesture": "Hand (RGB)", "ir": "IR reflector"},
                                value="gesture",
                            ).classes("q-mb-sm w-full justify-center").props(
                                "no-caps toggle-color=primary"
                            )

                        gesture_distance_tip = ui.column().classes("w-full q-mb-sm")
                        with gesture_distance_tip:
                            with ui.row().classes(
                                "baldi-distance-tip w-full items-start no-wrap"
                            ).style("gap: 10px;"):
                                ui.icon("straighten").classes("text-primary").style(
                                    "font-size: 1.5rem; opacity: 0.9;"
                                )
                                with ui.column().classes("gap-xs col"):
                                    ui.label("Best distance for hand tracking").classes(
                                        "text-caption text-weight-bold text-primary"
                                    )
                                    ui.label(
                                        "Hold your hand far enough from the webcam that your full hand "
                                        "fits comfortably in the frame — about arm’s length usually works best. "
                                        "Too close, and tracking often gets worse; a bit farther back is easier for the model."
                                    ).classes("text-body2 text-grey-8").style("line-height: 1.45;")

                        with ui.element("div").classes("baldi-video-wrap"):
                            gesture_display = ui.interactive_image().style(
                                "width:100%; height:auto; max-height:min(68vh, 720px); object-fit:contain; display:block; vertical-align:top;"
                            )
                            ir_display = None
                            if ir_tracker is not None:
                                ir_display = ui.interactive_image().style(
                                    "width:100%; height:auto; max-height:min(68vh, 720px); object-fit:contain; display:block; vertical-align:top;"
                                )

                        def sync_source_visibility():
                            global active_source
                            if source_toggle is not None and ir_display is not None:
                                active_source = source_toggle.value
                                gesture_display.set_visibility(active_source == "gesture")
                                ir_display.set_visibility(active_source == "ir")
                            else:
                                active_source = "gesture"
                            gesture_distance_tip.set_visibility(active_source == "gesture")

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

                        ui.label(
                            "Record saves MP4 to src/recordings/."
                        ).classes("text-caption text-grey-6 q-mt-sm").style("opacity: 0.95;")

                        with ui.row().classes("w-full justify-end q-mt-xs"):
                            record_button = ui.button("Record").props(
                                "unelevated no-caps rounded"
                            ).classes("text-weight-medium")
                            record_button.on_click(lambda: toggle_record(record_button))

                with ui.column().classes("col").style("flex: 0 1 400px; min-width: 240px;"):
                    with ui.card().classes("baldi-app-card baldi-app-card--accent q-pa-lg w-full"):
                        with ui.row().classes("items-baseline justify-between no-wrap q-mb-sm"):
                            ui.label("Recognition").classes(
                                "text-subtitle1 text-weight-medium text-grey-9"
                            )
                            ui.label("DTW").classes("baldi-pill")
                        ui.label(
                            "Templates use uppercase English letters A–Z only. Save your own strokes to improve matching."
                        ).classes("text-body2 text-grey-8 q-mb-md").style("line-height: 1.5;")

                        ui.separator().classes("q-my-sm bg-grey-4")

                        ui.label("How to draw").classes(
                            "text-caption text-weight-bold text-grey-7 text-uppercase q-mb-xs"
                        )
                        ui.label(
                            "Hand camera: other fingers in a loose fist; only thumb and index move. "
                            "Pinch to start a stroke, release to finish. Red dot while drawing, green when idle. "
                            "Stay far enough from the camera that your whole hand stays in frame (about arm’s length)."
                        ).classes("text-body2 text-grey-8 q-mb-sm").style("line-height: 1.45;")
                        if ir_tracker is not None:
                            ui.label(
                                "IR camera: hold the tip still briefly to start or stop a stroke. "
                                "Select IR above to evaluate those strokes."
                            ).classes("text-body2 text-grey-8 q-mb-md")

                        ui.label("Language").classes(
                            "text-caption text-weight-bold text-grey-7 text-uppercase q-mb-xs"
                        )
                        ui.toggle(["English", "Arabic"], value="English").classes("q-mb-sm")

                        label_input = ui.input("Letter label (optional)").props(
                            "clearable outlined dense hint='Uppercase A–Z only'"
                        ).classes("w-full q-mb-xs")

                        ui.label(
                            "Uppercase A–Z only for letter labels. Whatever you type is converted to uppercase for saving and matching."
                        ).classes("text-caption text-grey-7 q-mb-sm").style("line-height: 1.35;")

                        ui.label(
                            "Scores show template match (0–100%). Debug confidence is stored in the evaluation log."
                        ).classes("text-caption text-grey-6 q-mb-md")

                        predicted_label = ui.label("").classes(
                            "text-body1 whitespace-pre-line text-grey-9 q-mb-sm"
                        )
                        topk_label = ui.label("").classes(
                            "text-body2 text-grey-7 whitespace-pre-line q-mb-sm"
                        )
                        score_label = ui.label("").classes(
                            "text-body2 text-grey-7 whitespace-pre-line"
                        )

                        image_pred_label = ui.label("").classes("text-body2 text-grey-7")
                        if not SHOW_IMAGE_MODEL:
                            image_pred_label.set_visibility(False)

                        async def save_template():
                            raw = (label_input.value or "").strip()
                            label = raw.upper()
                            if raw:
                                label_input.value = label
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
                            .classes("w-full q-mb-sm")
                            .props("indeterminate color=primary")
                        )
                        eval_progress.set_visibility(False)

                        async def run_evaluation():
                            eval_btn.disable()
                            eval_progress.set_visibility(True)
                            try:
                                raw = (label_input.value or "").strip()
                                label = raw.upper()
                                if raw:
                                    label_input.value = label
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
                                        f"No template for '{label}'. Use an uppercase letter A–Z, or save your drawing as a template."
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

                        ui.separator().classes("q-my-sm")

                        with ui.column().classes("w-full q-gutter-xs"):
                            ui.button("Save as template", on_click=save_template).classes(
                                "w-full"
                            ).props("unelevated no-caps rounded color=primary")
                            eval_btn = ui.button("Evaluate", on_click=run_evaluation).classes(
                                "w-full"
                            ).props("unelevated no-caps rounded color=primary")
                            ui.button("Clear drawing", on_click=clear_drawing).classes(
                                "w-full"
                            ).props("outline no-caps rounded color=grey-8")

        with ui.tab_panel(previous_tab):
            with ui.column().classes("w-full baldi-page-shell").style("margin: 0 auto;"):
                with ui.card().classes("baldi-app-card q-pa-lg w-full"):
                    ui.label("History").classes("baldi-pill q-mb-xs")
                    ui.label("Evaluation log").classes(
                        "text-h6 text-weight-medium text-grey-9 q-mb-md"
                    )

                    @ui.refreshable
                    def records_view():
                        if not log_file.exists():
                            ui.label("No entries yet.").classes("text-body2 text-grey-7")
                            return

                        try:
                            lines = log_file.read_text().splitlines()
                        except Exception:
                            ui.label("Could not read log file.").classes("text-body2 text-grey-7")
                            return

                        if not lines:
                            ui.label("No entries yet.").classes("text-body2 text-grey-7")
                            return

                        with ui.scroll_area().classes("w-full").style("max-height: 320px;"):
                            with ui.column().classes("q-gutter-xs"):
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
                                        extra = (
                                            f"  img={img_pred} ({img_conf})" if img_pred is not None else ""
                                        )
                                        pred_extra = (
                                            f"  predicted={predicted} ({pred_conf:.2f})"
                                            if predicted is not None and pred_conf is not None
                                            else ""
                                        )
                                        top_extra = ""
                                        if isinstance(pred_top, list) and pred_top:
                                            try:
                                                top_extra = "  top=" + ",".join(
                                                    [
                                                        f"{t.get('label')}:{float(t.get('score') or 0.0):.2f}"
                                                        for t in pred_top[:3]
                                                    ]
                                                )
                                            except Exception:
                                                top_extra = ""
                                        ui.label(
                                            f"{ts}  label={lbl}  score={sc}  dist={dist}{pred_extra}{top_extra}{extra}"
                                        ).classes("text-caption text-grey-8 font-mono")
                                    except Exception:
                                        continue

                    ui.button("Refresh log", on_click=records_view.refresh).props(
                        "unelevated no-caps rounded outline color=grey-7"
                    ).classes("q-mb-sm")
                    records_view()

                    ui.separator().classes("q-mb-md bg-grey-4")

                    ui.label("Recordings").classes("baldi-pill q-mb-xs")
                    ui.label("Camera video files").classes(
                        "text-subtitle2 text-weight-medium text-grey-9 q-mb-md"
                    )

                    @ui.refreshable
                    def camera_recordings_view():
                        if not recordingsDir.exists():
                            ui.label("Recordings folder missing.").classes("text-body2 text-grey-7")
                            return
                        files = sorted(recordingsDir.glob("*.mp4"), key=lambda p: p.name)
                        if not files:
                            ui.label(
                                "No recordings yet. Use Record on the New recording tab."
                            ).classes("text-body2 text-grey-7")
                            return
                        with ui.column().classes("gap-sm"):
                            for f in files:
                                with ui.row().classes("items-center justify-between no-wrap"):
                                    ui.label(f.name).classes("text-body2 text-grey-8")
                                    ui.button("Download", on_click=lambda ff=f: ui.download(ff)).props(
                                        "flat dense no-caps color=primary"
                                    )

                    ui.button("Refresh list", on_click=camera_recordings_view.refresh).props(
                        "unelevated no-caps rounded outline color=grey-7"
                    )
                    camera_recordings_view()
