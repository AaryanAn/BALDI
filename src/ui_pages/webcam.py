import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path

import cv2
from nicegui import app, ui

from evaluation.letters import LetterEvaluator
from gestures.gestures import Gestures


srcDir = Path(__file__).resolve().parent.parent
path = str(srcDir / "gestures/hand_landmarker.task")
templates_dir = srcDir / "templates"
logs_dir = srcDir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "evaluations.jsonl"

cap = cv2.VideoCapture(0)
tracker = Gestures(path)
evaluator = LetterEvaluator(templates_dir)

latest_frame = None


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
    with ui.row().classes("items-start w-full gap-4 flex-wrap justify-center"):
        with ui.card().style("max-width: 800px; width: 100%;"):
            image = ui.interactive_image().style(
                "width:100%; height:auto; max-height:70vh; object-fit:contain;"
            )

            def update():
                if latest_frame:
                    image.set_source(f"data:image/jpeg;base64,{latest_frame}")

            ui.timer(0.03, update)

        with ui.card().style("min-width: 260px; max-width: 360px; width: 100%;"):
            ui.label("Welcome to BALDI Handwriting")

            label_input = ui.input("Letter label")
            score_label = ui.label("")

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

                result = evaluator.evaluate(label, traj)
                try:
                    record = {
                        "ts": datetime.utcnow().isoformat(),
                        "label": label,
                        "score": result.get("score"),
                        "distance": result.get("distance"),
                        "has_templates": result.get("has_templates"),
                        "num_templates": result.get("num_templates"),
                    }
                    with open(log_file, "a") as f:
                        f.write(json.dumps(record) + "\n")
                except Exception:
                    pass

                if not result["has_templates"]:
                    score_label.text = f"No templates for '{label}' yet"
                    ui.notify("Add at least one template first")
                    return

                score = result["score"]
                score_label.text = f"Score: {score:.2f}"
                ui.notify("Evaluation done")

            def clear_drawing():
                tracker.clear_path()
                ui.notify("Path cleared")

            ui.button("Save as template", on_click=save_template)
            ui.button("Evaluate", on_click=run_evaluation)
            ui.button("Clear Drawing", on_click=clear_drawing)
