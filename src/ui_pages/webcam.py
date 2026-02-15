import asyncio
import base64
import cv2

from nicegui import ui, app
from gestures.gestures import Gestures


cap = cv2.VideoCapture(0)
tracker = Gestures("gestures/hand_landmarker.task")

latest_frame = None


def process_frame():
    success, frame = cap.read()
    if not success:
        return None

    frame = cv2.flip(frame, 1)

    annotated_frame, fingertip = tracker.detect_index_fingertip(frame)

        # Draw path
    for i in range(1, len(tracker.path)):
        cv2.line(annotated_frame,
                tracker.path[i - 1],
                tracker.path[i],
                (0, 255, 255),
                3)

    
    _, buffer = cv2.imencode(
        ".jpg",
        annotated_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )

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
    with ui.row():
        with ui.card():

            image = ui.interactive_image().style(
                "height:90vh;"
            )

            def update():
                if latest_frame:
                    image.set_source(
                        f"data:image/jpeg;base64,{latest_frame}"
                    )

            ui.timer(0.03, update)
            
        
        with ui.card():
            ui.label("Welcome to BALDI Handwriting")
            
            def clear_drawing():
                tracker.clear_path()
                ui.notify('Path cleared!')
            
            ui.button('Clear Drawing', on_click= clear_drawing)