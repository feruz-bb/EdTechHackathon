# ClassVision

**Real-time classroom attention & mood analytics from a single webcam.**

![ClassVision demo](docs/demo.gif)

ClassVision watches a classroom through one ordinary webcam and tells the teacher, at a glance, who is focused and who is drifting — and *why*. It tracks every visible face, gives each student a persistent ID, measures where they are looking, whether they are talking, whether a phone is in their hand, and what mood they are in. All on-device, no cloud, no recording.

---

## Features

- **Multi-face tracking** — up to 10 simultaneous faces via MediaPipe Face Mesh.
- **Persistent per-person IDs** — each student gets a stable `Person #N` derived from a geometric face signature (no enrollment step, no photos stored).
- **Head pose** — yaw and pitch from facial landmarks, drawn as a purple direction arrow.
- **Iris-based gaze** — combined x/y gaze direction from both irises, drawn as yellow arrows.
- **Drowsiness detection** — Eye Aspect Ratio (EAR) flags closed/half-closed eyes.
- **Talking detection** — Mouth Aspect Ratio rolling stdev catches active speech.
- **Phone detection** — YOLOv8n picks up cell phones (COCO class 67) and flags students holding one near their body.
- **7-class mood** — ENGAGED / NEUTRAL / CONFUSED / UNHAPPY / SLEEPY / BORED / SURPRISED, smoothed over 15 frames.
- **Distraction score** — 30-frame rolling ratio combining gaze-off, eyes-closed, phone, talking, and bad-mood signals. Box turns red when a student crosses the threshold.
- **Live HUD** — focus count, focus %, visible vs registered totals, phone count, dominant class mood.

---

## Tech stack

- Python 3.12+
- [MediaPipe](https://developers.google.com/mediapipe) 0.10.14 — Face Mesh + iris landmarks
- [OpenCV](https://opencv.org/) — capture, drawing, PnP head pose
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — phone detection (`yolov8n.pt`)
- NumPy
- Packaged and run with [uv](https://docs.astral.sh/uv/)

---

## Setup

```bash
git clone <repo-url> ClassVision
cd ClassVision
uv sync
```

`yolov8n.pt` (~6.5 MB) is committed inside `classvision/`, so no separate model download is needed.

**macOS users:** grant your terminal Camera permission in *System Settings → Privacy & Security → Camera* before the first run.

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
2. **MediaPipe Face Mesh** finds up to 10 faces and 478 landmarks each.
3. For every face, ClassVision computes:
   - head pose (yaw, pitch),
   - gaze (x, y) from iris position,
   - EAR (eyes), MAR (mouth),
   - mood from mouth + brow + eyelid geometry.
4. A 30-frame rolling window aggregates per-frame "bad" flags into a **distraction ratio**. Above `0.5` → student is marked DISTRACTED with a reason (`PHONE`, `EYES OFF`, `HEAD DOWN`, `TALKING`, `EYES CLOSED`, `SLEEPY`, …).
5. The HUD aggregates everyone into a class-wide focus % and dominant mood.

### Tunable thresholds

All detection sensitivity lives at the top of `classvision/main.py` (lines 20–46). The most useful knobs:

| Constant | Default | Meaning |
|---|---:|---|
| `YAW_THRESHOLD` | 12 | degrees of head turn before "looking away" |
| `PITCH_DOWN_THRESHOLD` | 15 | degrees of head tilt down |
| `EAR_THRESHOLD` | 0.18 | eye openness below this = "eyes closed" |
| `MAR_STD_THRESHOLD` | 0.04 | mouth movement stdev for "talking" |
| `GAZE_X_THRESHOLD` | 0.35 | horizontal iris offset for "eyes off" |
| `GAZE_Y_THRESHOLD` | 0.45 | vertical iris offset for "eyes down/up" |
| `DISTRACTION_RATIO` | 0.5 | fraction of last 30 frames flagged → distracted |
| `SMILE_THRESHOLD` | 0.02 | mouth-corner lift for ENGAGED mood |
| `YAWN_MAR_THRESHOLD` | 0.55 | mouth-open ratio for BORED (yawn) |
| `FACE_MATCH_THRESHOLD` | 0.22 | signature distance for re-identifying a person |

Tune for your room lighting and camera angle — defaults assume a standard laptop webcam at desk distance.

---

## HUD legend

- **Yellow dots** — key landmarks (eye corners, nose tip, mouth corners, chin, forehead).
- **Purple arrow from nose** — head direction (yaw + pitch).
- **Yellow arrows from each iris** — gaze direction.
- **Green box** — student is FOCUSED.
- **Red box** — student is DISTRACTED (reason listed above the box).
- **Orange box labeled PHONE** — detected cell phone.
- **Text labels above each face** — white `Person #N`, mood label in its mood color, and a status line (`FOCUSED` or comma-separated reasons like `PHONE, EYES OFF, TALKING`).
- Per-person debug strip under each face: `yaw / pitch`, `gaze x / y`, `ear / mar`, `seen` (seconds), `dist` (%).

---

## Project layout

```
ClassVision/
├── README.md             ← you are here
├── pyproject.toml        ← uv workspace root
├── uv.lock
└── classvision/
    ├── main.py           ← the whole app (~450 lines, single file)
    ├── pyproject.toml    ← deps
    └── yolov8n.pt        ← YOLOv8 nano weights (committed)
```

---

## Known limitations

- Single webcam only (`cv2.VideoCapture(0)`). No video-file input, no IP camera, no multi-camera.
- No persistence across runs — when you quit, the `Person #N` registry is gone.
- No report export (CSV / HTML / dashboard) — everything is live on-screen.
- Mood and distraction thresholds are tuned for typical indoor lighting; very dim or very bright scenes need re-tuning.

---

## Credits

Built for a hackathon. Powered by MediaPipe, Ultralytics YOLOv8, and OpenCV.
