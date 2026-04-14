import sys

from nicegui import ui

try:
    import ui_pages.webcam
except AttributeError as e:
    err = str(e)
    if "is_repeated" in err or "FieldDescriptor" in err:
        print(
            "\nBALDI: MediaPipe/TensorFlow failed (common with conda `base` + wrong protobuf).\n"
            "  Use the project environment, then from `src/` run:\n"
            "    conda activate CIS4930\n"
            "    python main.py\n",
            file=sys.stderr,
        )
    raise


ui.run()