# ClassVision

**Real-time classroom attention and mood analytics from a single webcam, fully on-device.**

Built at the **Build with AI Hackathon** (SQB × New Uzbekistan University, Tashkent, 23–24 May 2026) by team **Spirit of TUIT**.

![ClassVision demo](docs/demo.gif)

ClassVision watches a classroom through one ordinary webcam and shows the teacher, at a glance, who is focused, who is drifting, and *why*. It tracks every visible face, gives each student a persistent ID, and measures where they are looking, whether their eyes are closed, whether they are talking, whether a phone is in their hand, and what mood they appear to be in. Everything runs locally: no cloud, no video recording, no stored photos.

---

## Features

- **Multi-face tracking**: up to 10 faces at once via MediaPipe Face Mesh (478 landmarks per face, including irises).
- **Persistent per-person IDs**: each student gets a stable `Person #N` from a geometric face signature. No enrollment step, no photos stored.
- **Head pose**: yaw and pitch estimated from facial-landmark geometry, drawn as a purple arrow.
- **Iris-based gaze**: x/y gaze direction from both irises, drawn as yellow arrows.
- **Eyes closed / drowsiness**: Eye Aspect Ratio (EAR).
- **Talking detection**: rolling standard deviation of the Mouth Aspect Ratio (MAR).
- **Phone detection**: YOLOv8n finds cell phones (COCO class 67) and flags a student when one is in their face/body zone.
- **7-class mood**: ENGAGED / NEUTRAL / CONFUSED / UNHAPPY / SLEEPY / BORED / SURPRISED, rule-based and smoothed over 15 frames.
- **Distraction score**: share of the last 30 frames flagged as distracted (eyes off, head turned, eyes closed, phone, talking, bored/sleepy). Above 50%, the box turns red and the reasons are shown.
- **Live HUD**: focused vs. total students, class focus %, visible vs. registered people, phone count, dominant class mood.

---

## Tech stack

- Python 3.12+
- [MediaPipe](https://developers.google.com/mediapipe) 0.10.14: Face Mesh with iris landmarks
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics): phone detection (`yolov8n.pt`)
- [OpenCV](https://opencv.org/): webcam capture and on-screen drawing
- NumPy: geometry (head pose, EAR/MAR, gaze ratios)
- [uv](https://docs.astral.sh/uv/): packaging and environment

---

## Setup

```bash
git clone https://github.com/feruz-bb/classvision.git
cd classvision
uv sync
```

`yolov8n.pt` (~6.5 MB) is committed inside `classvision/`, so no separate model download is needed.

**macOS users:** grant your terminal camera access in *System Settings → Privacy & Security → Camera* before the first run.

---

## Run

```bash
uv run python classvision/main.py
```

Press **ESC** to quit.

---

## How it works

Every frame:

1. **YOLOv8n** runs on every 2nd frame and locates cell phones.
2. **MediaPipe Face Mesh** finds up to 10 faces with 478 landmarks each.
3. For every face, ClassVision computes:
   - head pose (yaw, pitch) from nose, eye, chin and forehead landmark positions,
   - gaze (x, y) from iris position inside the eye,
   - EAR (eyes) and MAR (mouth),
   - mood from mouth, brow and eyelid geometry.
4. A 30-frame rolling window turns per-frame "bad" flags into a **distraction ratio**. Above `0.5`, the student is marked DISTRACTED with a reason (`PHONE`, `EYES OFF`, `HEAD DOWN`, `TALKING`, `EYES CLOSED`, `SLEEPY`, …).
5. The HUD aggregates everyone into a class-wide focus % and dominant mood.

### Tunable thresholds

All detection thresholds are constants at the top of `classvision/main.py`. The most useful ones:

| Constant | Default | Meaning |
|---|---:|---|
| `YAW_THRESHOLD` | 12 | degrees of head turn before "looking away" |
| `PITCH_DOWN_THRESHOLD` | 15 | degrees of head tilt down |
| `EAR_THRESHOLD` | 0.18 | eye openness below this = "eyes closed" |
| `MAR_STD_THRESHOLD` | 0.04 | mouth-movement stdev for "talking" |
| `GAZE_X_THRESHOLD` | 0.35 | horizontal iris offset for "eyes off" |
| `GAZE_Y_THRESHOLD` | 0.45 | vertical iris offset for "eyes down/up" |
| `DISTRACTION_RATIO` | 0.5 | fraction of last 30 frames flagged → distracted |
| `SMILE_THRESHOLD` | 0.02 | mouth-corner lift for ENGAGED mood |
| `YAWN_MAR_THRESHOLD` | 0.55 | mouth-open ratio for BORED (yawn) |
| `FACE_MATCH_THRESHOLD` | 0.22 | signature distance for re-identifying a person |

Defaults assume a standard laptop webcam at desk distance; tune them for your room's lighting and camera angle.

---

## HUD legend

- **Yellow dots**: key landmarks (eye corners, nose tip, mouth corners, chin, forehead).
- **Purple arrow from the nose**: head direction (yaw + pitch).
- **Yellow arrows from each iris**: gaze direction.
- **Green box**: student is FOCUSED.
- **Red box**: student is DISTRACTED (reasons listed above the box).
- **Orange box labeled PHONE**: detected cell phone.
- **Labels above each face**: `Person #N`, mood, and status (`FOCUSED` or reasons such as `PHONE, EYES OFF, TALKING`).
- **Debug lines under each face**: `yaw / pitch`, `gaze x / y`, `ear / mar`, seconds seen, distraction %.

---

## Privacy & responsible use

- Runs entirely on the local machine. Frames are processed in memory and are never saved or sent anywhere.
- No face images are stored. A person's ID comes from a 12-value geometric signature that exists only while the app is running and is discarded on exit.
- Mood and attention labels are rule-based estimates from face geometry, not a trained emotion model, and can be wrong (lighting, glasses, camera angle, individual differences).
- Intended as an aggregate signal that helps a teacher adjust a lesson, not for grading, discipline or monitoring individual students.

---

## Known limitations

- Single webcam only (`cv2.VideoCapture(0)`): no video-file input, IP camera or multi-camera setup.
- No persistence across runs: when you quit, the `Person #N` registry is gone.
- No report export (CSV / dashboard): everything is live on screen.
- Mood rules are hand-tuned and have not been validated on a labeled dataset.
- "Seconds seen" assumes ~30 FPS.

---

## Project layout

```
classvision/              ← repository root
├── README.md
├── pyproject.toml        ← uv workspace root
├── uv.lock
├── docs/
│   └── demo.gif          ← demo recording
└── classvision/
    ├── main.py           ← the whole app (~450 lines, single file)
    ├── pyproject.toml    ← dependencies
    └── yolov8n.pt        ← YOLOv8 nano weights (committed)
```

---

## Team

Team **Spirit of TUIT**, Build with AI Hackathon 2026

- **Feruzbek Baqoyev**: [your role, e.g. computer-vision pipeline and distraction logic]
- **[Name Surname]**: [role]
- **[Name Surname]**: [role]

## Credits

Built with MediaPipe, Ultralytics YOLOv8 and OpenCV.
