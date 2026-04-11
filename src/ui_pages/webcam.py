import asyncio
import base64
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
from nicegui import app, ui

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
        frame = process_frame()
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
                    predicted_label = ui.label("")
                    topk_label = ui.label("").classes("text-sm text-grey-7")
                    score_label = ui.label("").classes("text-sm text-grey-7")
                    image_pred_label = ui.label("").classes("text-sm text-grey-7")
                    if not SHOW_IMAGE_MODEL:
                        image_pred_label.set_visibility(False)

                    def save_template():
                        label = label_input.value or ""
                        paths = tracker.snapshot_paths()
                        traj = evaluator.get_trajectory(paths)
                        if traj.shape[0] == 0:
                            ui.notify("No path to save")
                            return

                        saved = evaluator.save_template(label, traj)
                        if saved is None:
                            ui.notify("Label is empty")
                            return

                        ui.notify(f"Saved template for '{label}'")

                    def run_evaluation():
                        label = label_input.value or ""
                        paths = tracker.snapshot_paths()
                        traj = evaluator.get_trajectory(paths)
                        if traj.shape[0] == 0:
                            ui.notify("No path to evaluate")
                            return

                        pred = evaluator.predict(traj, top_k=3)
                        if pred.get("predicted_label") is None:
                            conf = pred.get("confidence")
                            conf_txt = "" if conf is None else f" ({conf*100:.0f}%)"
                            predicted_label.text = f"Predicted: Uncertain{conf_txt}"
                        else:
                            conf = pred.get("confidence") or 0.0
                            predicted_label.text = f"Predicted: {pred['predicted_label']} ({conf*100:.0f}%)"

                        try:
                            top = pred.get("top") or []
                            if top:
                                items = [f"{t['label']} s={t['score']:.2f} d={t['distance']:.3f}" for t in top]
                                gap = None
                                if pred.get("best_distance") is not None and pred.get("second_distance") is not None:
                                    gap = float(pred["second_distance"] - pred["best_distance"])
                                gap_txt = "" if gap is None else f"  gap={gap:.3f}"
                                topk_label.text = "Top: " + ", ".join(items) + gap_txt
                            else:
                                topk_label.text = ""
                        except Exception:
                            topk_label.text = ""

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

                            if image_pred.get("available"):
                                pred = image_pred["predicted_label"]
                                conf = image_pred.get("confidence")
                                image_pred_label.text = f"Image model: {pred} ({conf:.2f})"
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
                            score_label.text = f"Similarity vs '{label}': {score:.2f}  ({n} template{'s' if n != 1 else ''})"
                        ui.notify("Evaluation done")

                    def clear_drawing():
                        tracker.clear_path()
                        ui.notify("Path cleared")

                    ui.button("Save as template", on_click=save_template)
                    ui.button("Evaluate", on_click=run_evaluation)
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
