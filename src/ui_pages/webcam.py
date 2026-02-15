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
        [int(cv2.IMWRITE_JPEG_QUALITY), 65],
    )

    return base64.b64encode(buffer).decode("utf-8")


async def background_capture():
    global latest_frame

    while True:
        frame = process_frame()
        if frame:
            latest_frame = frame

        await asyncio.sleep(0.03)


@app.on_startup
async def startup():
    asyncio.create_task(background_capture())


@app.on_shutdown
def shutdown():
    cap.release()


@ui.page("/")
def main_page():
    ui.query("body").style(
        "margin:0; padding:0; overflow:hidden; background:black;"
    )

    image = ui.interactive_image().style(
        "height:100vh;"
        "object-fit:cover;"
        "display:block;"
        "margin:auto;"
    )

    def update():
        if latest_frame:
            image.set_source(
                f"data:image/jpeg;base64,{latest_frame}"
            )

    ui.timer(0.05, update)


ui.run()
