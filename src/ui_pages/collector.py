import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path

import cv2
from nicegui import app, ui

from evaluation.letters import LetterEvaluator
from gestures.gestures import Gestures


src_dir = Path(__file__).resolve().parent.parent
model_path = str(src_dir / "gestures/hand_landmarker.task")
templates_dir = src_dir / "templates"
templates_dir.mkdir(parents=True, exist_ok=True)
logs_dir = src_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "collector_templates.jsonl"


# Reuse the same template location used by the main app,
# so collected samples immediately improve predictions.
collector_cap = cv2.VideoCapture(0)
collector_tracker = Gestures(model_path)
collector_evaluator = LetterEvaluator(templates_dir)

collector_latest_frame = None


def _collector_process_frame():
    success, frame = collector_cap.read()
    if not success:
        return None

    frame = cv2.flip(frame, 1)
    annotated_frame, _ = collector_tracker.detect_index_fingertip(frame)

    # Draw path
    for path in collector_tracker.paths:
        for i in range(1, len(path)):
            cv2.line(
                annotated_frame,
                path[i - 1],
                path[i],
                (0, 255, 255),
                3,
            )

    _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode("utf-8")


async def _collector_background_capture():
    global collector_latest_frame

    while True:
        frame = _collector_process_frame()
        if frame:
            collector_latest_frame = frame
        await asyncio.sleep(0.02)


@app.on_startup
async def collector_startup():
    asyncio.create_task(_collector_background_capture())


@app.on_shutdown
def collector_shutdown():
    collector_cap.release()


@ui.page("/collector")
def collector_page():
    with ui.header().classes("bg-primary items-center justify-between"):
        ui.label("BALDI Template Collector").classes("text-h5 font-bold")

    with ui.row().classes("items-start w-full gap-4 flex-wrap justify-center"):
        with ui.card().style("max-width: 900px; width: 100%;"):
            image = ui.interactive_image().style(
                "width:100%; height:auto; max-height:75vh; object-fit:contain;"
            )

            def update():
                if collector_latest_frame:
                    image.set_source(f"data:image/jpeg;base64,{collector_latest_frame}")

            ui.timer(0.03, update)

        with ui.card().style("min-width: 260px; max-width: 420px; width: 100%;"):
            ui.label(
                "Flow: 1) Type the letter, 2) Draw it with your finger, 3) Confirm or redo."
            ).classes("text-sm text-grey-7")

            contributor_input = ui.input("Your name (optional)")
            label_input = ui.input("Letter to record (A–Z)").props("clearable")
            step_label = ui.label("Step 1: enter the letter you want to record.").classes(
                "text-sm text-grey-7"
            )
            info_label = ui.label("").classes("text-sm text-grey-7")

            current = {"label": None, "traj": None}

            def clear_paths():
                collector_tracker.clear_path()
                current["traj"] = None
                info_label.text = ""
                ui.notify("Cleared current drawing")

            def set_letter():
                raw = (label_input.value or "").strip()
                if not raw:
                    ui.notify("Please enter a letter.")
                    return
                if len(raw) != 1:
                    ui.notify("Please enter exactly one character (e.g. A).")
                    return
                # We treat upper/lower case the same; store as uppercase.
                label = raw.upper()
                label_input.value = label
                current["label"] = label
                current["traj"] = None
                clear_paths()
                step_label.text = f"Step 2: draw the letter '{label}' in the webcam area, then click 'Done drawing'."

            def done_drawing():
                label = current["label"]
                if not label:
                    ui.notify("Set the letter first.")
                    return

                paths = collector_tracker.snapshot_paths()
                traj = collector_evaluator.get_trajectory(paths)
                if traj.shape[0] == 0:
                    ui.notify("No drawing captured yet.")
                    return

                current["traj"] = traj
                info_label.text = f"Preview ready for '{label}'. If it looks good, click 'Save this sample'. Otherwise click 'Redo drawing'."
                step_label.text = f"Step 3: confirm or redo the letter '{label}'."

            def save_sample():
                label = current["label"]
                traj = current["traj"]
                if not label:
                    ui.notify("Set the letter first.")
                    return
                if traj is None or traj.shape[0] == 0:
                    ui.notify("Draw the letter and click 'Done drawing' first.")
                    return

                fname = collector_evaluator.save_template(label, traj)
                if fname is None:
                    ui.notify("Could not save template")
                    return

                contributor = (contributor_input.value or "").strip() or None
                try:
                    record = {
                        "ts": datetime.utcnow().isoformat(),
                        "label": label,
                        "template_path": str(fname),
                        "contributor": contributor,
                    }
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                except Exception:
                    pass

                info_label.text = f"Saved sample for '{label}' → {fname.name}"
                ui.notify(f"Saved sample for '{label}'")
                # Keep the same letter so the user can record multiple samples quickly.
                current["traj"] = None
                collector_tracker.clear_path()
                step_label.text = f"Step 2: draw another '{label}' or change the letter."

            ui.button("Set letter", on_click=set_letter)
            ui.button("Done drawing", on_click=done_drawing)
            ui.button("Save this sample", on_click=save_sample)
            ui.button("Redo drawing", on_click=clear_paths).props("flat")

