import asyncio
import base64
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
from nicegui import app, run, ui

from evaluation.letters import LetterEvaluator
from gestures.gestures import Gestures


srcDir = Path(__file__).resolve().parent.parent
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

latest_frame = None
SHOW_IMAGE_MODEL = os.getenv("BALDI_SHOW_IMAGE_MODEL", "").strip() in {"1", "true", "True", "yes", "YES"}


def process_frame():
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

    
    _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    return base64.b64encode(buffer).decode("utf-8")


async def background_capture():
    global latest_frame

    while True:
        # MediaPipe/OpenCV off the asyncio loop so WebSocket heartbeats are not starved.
        frame = await run.io_bound(process_frame)
        if frame:
            latest_frame = frame

        await asyncio.sleep(0.02)


@app.on_startup
async def startup():
    asyncio.create_task(background_capture())


@app.on_shutdown
def shutdown():
    cap.release()


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
                    image = ui.interactive_image().style(
                        "width:100%; height:auto; max-height:75vh; object-fit:contain;"
                    )

                    def update():
                        if latest_frame:
                            image.set_source(f"data:image/jpeg;base64,{latest_frame}")

                    ui.timer(0.03, update)

                with ui.card().style("min-width: 260px; max-width: 420px; width: 100%;"):
                    ui.label("Built-in templates for A–Z, a–z. You can save your own for better matching.").classes(
                        "text-sm text-grey-7"
                    )

                    ui.label("Please select your language:")
                    ui.toggle(["English", "Arabic"], value="English")

                    label_input = ui.input("Letter label (optional)").props("clearable")
                    ui.label(
                        "Confidence = how sure the app is that one letter wins over the others. "
                        "Template match = how similar your stroke is to saved examples of a letter."
                    ).classes("text-caption text-grey-6")
                    predicted_label = ui.label("").classes("text-body1 whitespace-pre-line")
                    topk_label = ui.label("").classes("text-sm text-grey-7 whitespace-pre-line")
                    score_label = ui.label("").classes("text-sm text-grey-7 whitespace-pre-line")
                    image_pred_label = ui.label("").classes("text-sm text-grey-7")
                    if not SHOW_IMAGE_MODEL:
                        image_pred_label.set_visibility(False)

                    async def save_template():
                        label = label_input.value or ""
                        paths = tracker.snapshot_paths()

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
                            paths = tracker.snapshot_paths()

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

                            conf = pred.get("confidence")
                            conf_pct = None if conf is None else f"{conf * 100:.0f}%"
                            top = pred.get("top") or []
                            best_guess = top[0]["label"] if top else None

                            if pred.get("predicted_label") is None:
                                if best_guess is not None and conf_pct is not None:
                                    predicted_label.text = (
                                        f"Best match by shape: “{best_guess}” — not confident enough to lock in.\n"
                                        f"Confidence in one clear winner (vs other letters): {conf_pct}. "
                                        f"That usually means several letters fit your stroke almost as well."
                                    )
                                elif conf_pct is not None:
                                    predicted_label.text = (
                                        f"Could not pick a clear letter.\n"
                                        f"Confidence in one winner: {conf_pct}."
                                    )
                                else:
                                    predicted_label.text = "Could not compare to templates."
                            else:
                                letter = pred["predicted_label"]
                                c = pred.get("confidence") or 0.0
                                predicted_label.text = (
                                    f"Best guess: “{letter}”\n"
                                    f"Confidence it’s this letter (compared to all other letters): {c * 100:.0f}%"
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
                        tracker.clear_path()
                        ui.notify("Path cleared")

                    ui.button("Save as template", on_click=save_template)
                    eval_btn = ui.button("Evaluate", on_click=run_evaluation)
                    ui.button("Clear Drawing", on_click=clear_drawing)

        with ui.tab_panel(previous_tab):
            ui.label("Previous recordings")

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

            ui.button("Refresh", on_click=records_view.refresh)
            records_view()
