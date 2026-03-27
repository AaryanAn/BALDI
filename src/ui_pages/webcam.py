import asyncio
import base64
import cv2
import os

from nicegui import ui, app
from gestures.gestures import Gestures

from pathlib import Path


recordingsDir = Path(__file__).resolve().parent.parent / "recordings"
recordingsDir.mkdir(exist_ok=True)
srcDir = Path(__file__).resolve().parent.parent
path = str(srcDir / "gestures/hand_landmarker.task")

cap = cv2.VideoCapture(0)
tracker = Gestures(path)

width = 0
height = 0
raw_frame = None
latest_frame = None
is_recording = False
video_writer = None
image = None


def process_frame():
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

    raw_frame = annotated_frame.copy()
    
    _, buffer = cv2.imencode(
        ".jpg",
        annotated_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )

    return base64.b64encode(buffer).decode("utf-8")


async def background_capture():
    global latest_frame, video_writer, is_recording, raw_frame

    while True:
        frame = process_frame()
        if frame:
            latest_frame = frame

        if frame and raw_frame is not None and is_recording and video_writer:
            video_writer.write(raw_frame)

        await asyncio.sleep(0.01)

def toggle_record(record_button):
    global is_recording, video_writer, width, height
    if is_recording:
        is_recording = False
        video_writer.release()
        video_writer = None
        record_button.props("color=primary")
    else:
        recordingFile = str(recordingsDir / f"recording_{len(os.listdir(recordingsDir))}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(recordingFile, fourcc, 20.0, (width, height))
        is_recording = True
        record_button.props("color=negative")


@app.on_startup
async def startup():
    global width, height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    asyncio.create_task(background_capture())


@app.on_shutdown
def shutdown():
    cap.release()


@ui.page("/")
def main_page():
    with ui.header().classes('bg-primary items-center justify_between'):
        ui.label("Welcome to BALDI Handwriting").classes("text-h5 font-bold items-center justify-between")
        
        with ui.tabs().classes("absolute-center") as tabs: 
            recordTab = ui.tab("New Recording")
            previousRecordingsTab = ui.tab("Previous Recordings")


    with ui.tab_panels(tabs, value=recordTab).classes("w-full"):
        with ui.tab_panel(recordTab):
            with ui.row():
                with ui.card().classes("w-full justify-center items-center"):

                    display = ui.interactive_image().style(
                        "height:70vh; width: auto; background: #000;"
                    )
                    ui.timer(0.03, lambda: display.set_source(f"data:image/jpeg;base64,{latest_frame}") if latest_frame else None)
                
                record_button = ui.button("Record")
                record_button.on_click(lambda: toggle_record(record_button))

                def clear_drawing():
                    tracker.clear_path()
                    ui.notify('Path cleared!')           
                ui.button('Clear Drawing', on_click= clear_drawing)

                with ui.card():
                    ui.label("Please select your language:")
                    ui.toggle(["English", "Arabic"], value="English")
        
        with ui.tab_panel(previousRecordingsTab):
            ui.label("Saved Gestures").classes("text-h6")
            # Refresh list when tab is clicked
            def refresh_list():
                list_container.clear()
                with list_container:
                    files = sorted(recordingsDir.glob("*.mp4"))
                    for f in files:
                        with ui.row().classes('items-center'):
                            ui.label(f.name)
                            ui.button(icon='download', on_click=lambda f=f: ui.download(f))
            
            list_container = ui.column()
            ui.button("Refresh List", on_click=refresh_list)
            ui.timer(0.1, refresh_list, once=True)