## BALDI Template Collector

This is a minimal tool for collecting air-drawn letter templates from your team to improve DTW-based predictions.

It uses the same webcam + fingertip tracking as the main app, but with a **very simple flow**:

1. **Type the letter you want to record**
2. **Draw it with your finger**
3. **Confirm or redo**, then move on

All collected samples are stored under the shared `src/templates/` directory, so the main BALDI app can use them immediately.

---

## Environment

Use the same environment as the main BALDI app.

From the repo root:

```bash
conda activate CIS4930          # or the environment you already use for BALDI
pip install -r requirements.txt
```

You only need to do this once per machine.

---

## How to launch the collector

From the repo root:

```bash
cd src
python main_collect.py
```

- Your browser should open automatically (or visit `http://localhost:8080/collector`).
- You will see the **BALDI Template Collector** header and the webcam feed.

---

## Collector flow (for team members)

1. **Set the letter**
   - In the right panel, find the field **“Letter to record (A–Z)”**.
   - Type a **single letter** (e.g. `A`, `b`).
   - Click **“Set letter”**.
   - The app normalizes this to **uppercase internally** (so `a` and `A` are treated as the same class).

2. **Draw the letter**
   - After you click “Set letter”, the instructions will say:
     - *“Step 2: draw the letter 'X' in the webcam area, then click 'Done drawing'.”*
   - Use your fingertip to draw the letter in the webcam window.
   - When you are satisfied with the stroke, click **“Done drawing”**.

3. **Confirm or redo**
   - After “Done drawing”, the instructions change to:
     - *“Preview ready for 'X'. If it looks good, click 'Save this sample'. Otherwise click 'Redo drawing'.”*
   - If the drawing is good:
     - Click **“Save this sample”**.
     - The trajectory is:
       - normalized via the same preprocessing as the main app
       - saved to `src/templates/X/<n>.npy`
       - logged to `logs/collector_templates.jsonl` with timestamp, label, and your (optional) name
   - If the drawing is bad:
     - Click **“Redo drawing”** to clear it and try again **for the same letter**.

4. **Collect multiple samples**
   - After saving a sample, the same letter remains selected.
   - To record more examples of that letter:
     - draw again
     - click **“Done drawing”**, then **“Save this sample”**
   - To switch letters:
     - type a new letter in **“Letter to record (A–Z)”**
     - click **“Set letter”**

---

## What gets stored where

- **Templates**:
  - Path: `src/templates/<LETTER>/<index>.npy`
  - These are the normalized trajectories used directly by the DTW evaluator.

- **Collector log**:
  - Path: `logs/collector_templates.jsonl`
  - Each line is a JSON object:
    - `ts`: timestamp
    - `label`: the letter (uppercase)
    - `template_path`: path to the saved `.npy`
    - `contributor`: optional name from the UI

Git is configured **not** to commit the `logs/` directory or large artifacts by default.

---

## How this improves predictions

The main BALDI app uses a **DTW-based auto-prediction** over all templates in `src/templates/`. By collecting:

- multiple samples per letter
- from multiple people

you:

- give DTW more realistic trajectories for each class
- increase the separation between the best-matching letter and the runner-up
- directly improve the **“Predicted: X (Y%)”** behavior in the main app

No retraining step is required—just relaunch the main app after pulling the updated templates.

