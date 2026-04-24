# BALDI — Final Project Report

**Course:** CIS 4914 Senior Project  
**Project name:** BALDI (Handwriting in the air)  
**Date:** _[Fill in]_  

---

## Title Page

- **Title**: BALDI — Handwriting in the Air (Trajectory Capture + Template Matching)
- **Team name**: _[Fill in]_
- **Team members**: _[Fill in names]_
- **Course name**: CIS 4914 Senior Project
- **Advisor**: _[Fill in]_
- **Advisor e-mail**: _[Fill in]_
- **YouTube link to Final Presentation**: _[Fill in]_
- **Keywords**:
  - air-writing
  - computer vision
  - MediaPipe
  - trajectory normalization
  - dynamic time warping (DTW)
  - template matching
  - NiceGUI
  - React + Vite (prototype)
  - IR tip tracking

---

## Project Overview

### Project Description
BALDI is a vision-based system for capturing **free-space (“air”) handwriting trajectories** and evaluating them against known reference templates. Users draw letters in the air using either:

- **RGB hand tracking** (index fingertip tracking with pinch-to-draw interaction), and optionally
- **IR reflector tip tracking** (for more robust, high-contrast tip localization when a second IR camera is available).

The project reconstructs 2D trajectories from video, normalizes them to reduce variability (scale/translation/rotation), and compares them to stored templates using **Dynamic Time Warping (DTW)** to support recognition and evaluation.

### Goals and Objectives
- Capture usable air-writing trajectories from commodity cameras (webcam; optional IR camera).
- Normalize trajectories so comparisons are meaningful across users and drawing scale.
- Evaluate similarity against templates and provide user-facing feedback (score + best guess).
- Support iterative improvement by letting users **save additional templates** (no retraining required for DTW).
- Provide a lightweight path to collect team templates at scale (template-collector workflow).
- Provide a browser-only prototype to demonstrate the same DTW/template approach in JavaScript.

### Scope
**In-scope**
- Trajectory capture from video, stroke accumulation, and clearing/saving/evaluating strokes
- Template storage on disk as `.npy` (Python) and in browser storage (JavaScript prototype)
- DTW-based similarity scoring and “best guess” prediction
- Optional IR-based tip tracking pipeline
- A minimal automated test suite for key algorithms and behaviors

**Out-of-scope / not emphasized**
- A production-grade database and user accounts
- Large-scale model training pipeline as the primary evaluation method (DTW is the main evaluation approach)
- Formal deployment to cloud infrastructure (project runs locally)

### Team Members and Roles/Responsibilities
_[Fill in each member’s name and responsibilities. Example:]_
- **Member 1**: Tracking pipeline, UI integration, evaluation logic
- **Member 2**: Template tooling + dataset utilities + testing
- **Member 3**: Web prototype (React/Vite), MediaPipe JS integration, UX
- **Member 4**: IR tracking experiments, hardware integration, tuning

---

## Deliverables

### Completed Deliverables (Repo Evidence)
- **Main BALDI application (Python + NiceGUI)**: interactive UI for drawing, saving templates, and evaluating strokes.
  - Entry point: `src/main.py`
  - UI pages: `src/ui_pages/webcam.py`, supporting modules under `src/ui_pages/`
- **Template Collector (Python + NiceGUI)**: streamlined flow for teammates to record many templates quickly.
  - Entry point: `src/main_collect.py`
  - Collector page: `src/ui_pages/collector.py`
  - Output: templates saved under `src/team_templates/<LETTER>/<index>.npy`
- **DTW evaluation and prediction**:
  - DTW implementation: `src/evaluation/dtw.py`
  - Template manager and evaluator: `src/evaluation/letters.py`
- **Trajectory preprocessing**:
  - Flattening, resampling, normalization: `src/trajectory/normalization.py`
- **Optional IR tip tracking**:
  - IR tracker implementation: `src/tracking/ir_tracker.py`
  - Optional runtime integration in the main UI (second camera index)
- **Browser-only JavaScript prototype (React + TypeScript + Vite)**:
  - App: `web/src/App.tsx`
  - MediaPipe Tasks Vision hand landmarking: `web/src/lib/handTracker.ts`
  - Template storage + import bridge: `web/src/lib/templates.ts`
- **Automated tests (pytest)**:
  - `tests/test_autopredict.py`, `tests/test_tracking.py`, and other `tests/test_*.py`

### Functional Specifications (Implemented)
- **Trajectory capture**: Build stroke paths from fingertip/tip detections over time.
- **Trajectory normalization**:
  - Translation to centroid
  - Scale to unit size
  - Rotation alignment using principal axis (PCA-like eigenvector on covariance)
  - Resampling to a fixed number of points (default 100 in Python)
- **Evaluation**: Compute DTW distance between an input trajectory and stored templates for a label; convert distance to a score.
- **Prediction (“best guess”)**: Compare a trajectory to all saved templates, return best label when separation is sufficient; otherwise return “uncertain”.
- **Template creation**: Save a normalized trajectory as a new template for a label.

### Current Project Status
- **Status**: Working end-to-end local demos for (a) Python NiceGUI app and (b) JS prototype; IR tracking is supported when hardware is available.
- **Known remaining work/backlog (examples to tailor)**:
  - Improve robustness under poor lighting / occlusion (especially RGB-only mode)
  - Improve stroke segmentation and user guidance for consistent stroke start/stop
  - Expand beyond A–Z (e.g., Arabic support) with curated templates and UI toggles fully wired into evaluation
  - Broaden test coverage for UI-facing utilities and error handling

### Known Issues and Risks
- **Hardware variability**: Camera quality and frame rate can affect tracking stability.
- **Lighting/occlusion sensitivity**: RGB hand tracking may degrade if the hand is too close or partially out of frame.
- **Template quality**: DTW performs best when templates are representative and plentiful; poor templates lead to ambiguous matches.
- **Second camera availability**: IR mode depends on a second camera and physical IR reflector/tip setup.

---

## Technical Details

### Codebase Overview
- **Repository**: Single git repository (this repo root).
- **Version control**: Git.
- **Remote**: GitHub remote configured as `origin`.
- **Branching**:
  - Work appears on feature branches (example currently checked out: `JS-integration/ml-with-ir-camera`).
  - Template collection workflow uses personal branches (see `TEMPLATE_COLLECTOR_TEAM.md`).

### Key Dependencies and Libraries

**Python application**
- **NiceGUI** (`nicegui`): web UI served from Python.
- **OpenCV** (`opencv-python`, `opencv-contrib-python`): camera access, frame processing, drawing overlays, video recording.
- **MediaPipe** (`mediapipe`): hand landmarking model support in the Python pipeline (via the `hand_landmarker.task` file referenced by the app).
- **NumPy / SciPy**: numeric operations, trajectory representation.
- **TensorFlow / Keras**: included as dependencies (image model support is optional and guarded behind an environment flag in the UI).

**JavaScript prototype**
- **React + ReactDOM**: UI layer.
- **Vite**: development/build toolchain.
- **@mediapipe/tasks-vision**: browser hand landmarking.
- **npyjs**: reading `.npy` data (used by template tooling/import workflows).

### Deployment / Run Instructions (Local)

#### Python (Main BALDI App)
From repo root (one-time setup):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the app:

```bash
cd src
python main.py
```

Notes:
- The UI is served by NiceGUI (default local address printed in the console).
- The app attempts to open the default webcam at index 0.
- The app optionally tries to open an IR camera at index 1; if unavailable, IR mode is disabled automatically.

#### Python (Template Collector)

```bash
cd src
python main_collect.py
```

Then visit `http://localhost:8080/collector` if it does not open automatically.

#### JavaScript Prototype (Web)

```bash
cd web
npm install
npm run dev
```

Notes:
- Uses browser webcam permission flow.
- Uses CDN-hosted WASM assets and a CDN-hosted `hand_landmarker.task` model (see `web/src/lib/handTracker.ts`).

### Configuration
- **Environment flags**:
  - `BALDI_SHOW_IMAGE_MODEL`: when set to a truthy value, the Python UI attempts to load and display an image-model prediction in addition to DTW results.

### Database Schema and Data Migration
This project intentionally avoids a traditional database. Data is stored as files on disk and in local browser storage.

**Python data storage**
- **Templates**: `src/templates/<LABEL>/<index>.npy` (main app); `src/team_templates/<LABEL>/<index>.npy` (collector workflow).
- **Template metadata**: `src/templates/<LABEL>/<index>.json` (written alongside `.npy` by `LetterEvaluator.save_template`).
- **Evaluation logs**: `src/logs/evaluations.jsonl` (JSONL: one record per evaluation run).
- **Collector logs**: `src/logs/collector_templates.jsonl` (JSONL: timestamp, label, path, contributor name).
- **Recordings**: `src/recordings/*.mp4` (optional recordings from the UI).

**JavaScript prototype storage**
- **Templates**: stored in `localStorage` under key `baldi.templates.v1`.
- **Optional import of Python templates**: fetches `/templates.json` if served by the hosting environment (see `web/src/lib/templates.ts`).

Because file-based storage is used, there is no DB migration process; changes are managed by git history and/or by regenerating templates as needed.

### Software Architecture

#### High-Level Pipeline (Python App)
1. **Camera capture** (`cv2.VideoCapture`)
2. **Tracking**
   - RGB mode: hand landmarking (via the `Gestures` module) produces fingertip positions and stroke state
   - Optional IR mode: `IRTracker` performs background subtraction + blob selection + proximity gating + smoothing
3. **Stroke collection**: paths are accumulated into lists of (x, y) points
4. **Preprocessing** (`preprocess_paths`)
   - flatten → resample → normalize
5. **Evaluation / Prediction**
   - compare to stored templates using DTW distance
   - convert distance to score; compute top-K matches; optionally return “uncertain”
6. **UI**
   - renders live preview + overlaid paths
   - provides “Save as template”, “Evaluate”, “Clear drawing”, and history/log views

#### Key Modules (Python)
- **UI**: `src/ui_pages/webcam.py`, `src/ui_pages/collector.py`
- **Tracking**: `src/gestures/gestures.py`, `src/tracking/ir_tracker.py`
- **Trajectory**: `src/trajectory/normalization.py`
- **Evaluation**: `src/evaluation/dtw.py`, `src/evaluation/letters.py`

#### High-Level Pipeline (Web Prototype)
1. `getUserMedia()` webcam stream in browser
2. MediaPipe Tasks Vision detects hand landmarks (`HandLandmarker.detectForVideo`)
3. A drawing session collects trajectories based on pinch logic
4. Trajectory normalization + DTW against saved templates
5. Templates persist in browser local storage; optional merge with Python-exported templates

---

## Testing Information

### Summary of Testing Conducted
- **Unit tests** (pytest):
  - DTW correctness checks (identical vs shifted)
  - Trajectory preprocessing behavior
  - Template saving/loading and evaluation score bounds
  - Autopredict behavior for clear vs ambiguous winners

### How to Run Tests

```bash
python -m pytest -q
```

Environment note:
- Some tests import `mediapipe` (via `gestures.gestures`). If your Python environment has an incompatible `protobuf`/TensorFlow combination (common in conda `base`), test collection may fail with errors like `FieldDescriptor ... has no attribute 'is_repeated'`.
- Use the same project environment you run the app with (for example, the repo’s documented `CIS4930` conda env) rather than conda `base`.

### Test Cases and Results (Representative)
Examples from the repository tests:
- **DTW distance sanity**: identical trajectories produce distance 0; shifted trajectories produce larger distance.
- **Evaluator flow**: saving a template, then evaluating the same trajectory returns `has_templates=True` and score in \([0, 1]\).
- **Autopredict**:
  - returns a predicted label when there is a clear winner
  - returns `predicted_label=None` when ambiguity is forced via identical templates and a larger `gap_min_dist`

### Outstanding Bugs / Defect Log
No formal defect tracking database is included in-repo. Known issues are tracked informally via:
- the `docs/engineering_journal.md` notes,
- git commit history, and
- observed runtime issues (hardware variance, lighting, etc.).

_[Optionally add a table here if your team maintained a bug list:]_
- **ID** | **Description** | **Severity** | **Status** | **Workaround**

---

## User Documentation

### Who This Is For
- Students/instructors/demo viewers who want to see end-to-end air-writing capture and DTW-based matching.
- Team members collecting template samples to improve recognition accuracy.

### Quick Start (Main App)
1. Install Python dependencies (see Deployment / Run Instructions).
2. Run `python main.py` from `src/`.
3. In the UI:
   - **Draw** using the hand camera mode (pinch to start, release to stop).
   - **Evaluate** to see the best guess and top matches.
   - (Optional) enter a label and click **Save as template** to add examples.
   - Use **History** to view recent evaluation log entries and download recordings.

### Template Collection Workflow (Team)
Use the Template Collector tool:
1. Run `python main_collect.py` from `src/`.
2. Visit `/collector`.
3. For each letter:
   - Set the letter (A–Z).
   - Draw the letter.
   - Confirm and save.
4. Commit templates from `src/team_templates/` on your personal branch.

References:
- `docs/TEMPLATE_COLLECTOR.md`
- `TEMPLATE_COLLECTOR_TEAM.md`

### JavaScript Prototype (Browser)
1. `cd web && npm install && npm run dev`
2. Allow camera permissions in the browser.
3. Pinch-to-draw, then “Evaluate” for DTW matching.
4. “Save as template” stores the current trajectory as an example for the selected label.

### FAQs / Troubleshooting
- **The camera feed is blank**: verify OS/browser camera permissions (web) or that OpenCV can open the device index (python).
- **Tracking is unstable**: move your hand farther back so your full hand stays in frame; avoid harsh backlighting.
- **Predictions are ambiguous**: collect more templates per letter from multiple users; ambiguous DTW matches are expected with sparse templates.
- **IR mode doesn’t appear**: the app disables IR mode when camera index 1 cannot be opened; verify hardware and camera indices.

---

## Transition Plan / Next Steps

### Knowledge Transfer
- Walkthrough of key modules:
  - Tracking (RGB vs IR) and how strokes are accumulated
  - Normalization and DTW scoring
  - Template storage layout and the collector workflow
- Ensure a new contributor can:
  - set up the Python environment and run `src/main.py`
  - run the collector and commit team templates
  - run tests and interpret failures

### Sustainability Plan
- Maintain a curated canonical template set in `src/templates/` for demos.
- Keep template collection on personal branches, and periodically curate/merge.
- Add CI (future) to run `pytest` and basic web lint/build checks.
- Add a small export step for Python templates (`templates.json`) so the JS prototype can reliably import them.

---

## Acknowledgments
_[Fill in]_  
Examples:
- Project advisor for guidance and regular feedback
- Any lab or organization providing equipment (e.g., IR camera)
- Any scholarship, fellowship, or sponsor support

---

## Appendices

### Appendix A — Key Code Fragments (Pointers)
- **DTW distance**: `src/evaluation/dtw.py`
- **Template evaluator + autopredict**: `src/evaluation/letters.py`
- **Normalization pipeline**: `src/trajectory/normalization.py`
- **IR tracking algorithm**: `src/tracking/ir_tracker.py`
- **NiceGUI main UI page**: `src/ui_pages/webcam.py`
- **Collector page**: `src/ui_pages/collector.py`
- **Web prototype hand tracking**: `web/src/lib/handTracker.ts`

### Appendix B — Result Artifacts Produced by the App
- `src/logs/evaluations.jsonl`
- `src/logs/collector_templates.jsonl`
- `src/recordings/*.mp4`
- `src/templates/<LABEL>/*.npy` and `src/team_templates/<LABEL>/*.npy`

---

## Biography

_[Add one biography per team member. Suggested template:]_

### Team Member: _[Name]_
- **Background**: _[prior internships/roles/projects]_
- **Contributions to BALDI**: _[what you worked on]_
- **Post-graduation goals**: _[industry role / graduate study]_
- **Personal interests**: _[1 sentence]_

