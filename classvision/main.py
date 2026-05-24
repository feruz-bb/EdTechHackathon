from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

# --- Модели ---
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

yolo = YOLO(str(Path(__file__).parent / "yolov8n.pt"))
PHONE_CLASS_ID = 67

# --- Пороги ---
YAW_THRESHOLD        = 12
PITCH_DOWN_THRESHOLD = 15
PITCH_UP_THRESHOLD   = -15
EAR_THRESHOLD        = 0.18
MAR_STD_THRESHOLD    = 0.04
MAR_MAX_THRESHOLD    = 0.05
GAZE_X_THRESHOLD     = 0.35
GAZE_Y_THRESHOLD     = 0.45
DISTRACTION_RATIO    = 0.5

# --- Mood ---
SMILE_THRESHOLD     = 0.02
FROWN_THRESHOLD     = -0.01
BROW_FURROW_THRESH  = 0.27
BROW_RAISE_THRESH   = 0.18
YAWN_MAR_THRESHOLD  = 0.55
SLEEPY_EAR_THRESH   = 0.21

# --- Face Recognition ---
FACE_MATCH_THRESHOLD = 0.22
IOU_MATCH_THRESHOLD  = 0.3
MAX_ABSENCE_FRAMES   = 150

GHOST_THRESHOLD_SEC  = 3.0    # entries seen less than this = ghosts, will be cleaned up
PANEL_RECENT_SEC     = 30.0   #

# --- Landmarks ---
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_EYE_CORNERS  = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_VERT     = (159, 145)
RIGHT_EYE_VERT    = (386, 374)
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

MOUTH_LEFT       = 61
MOUTH_RIGHT      = 291
MOUTH_TOP        = 13
MOUTH_BOTTOM     = 14
UPPER_LIP_CENTER = 0
LOWER_LIP_CENTER = 17
LEFT_BROW_INNER  = 55
RIGHT_BROW_INNER = 285
LEFT_BROW_OUTER  = 105
RIGHT_BROW_OUTER = 334
LEFT_EYE_TOP_REF  = 159
RIGHT_EYE_TOP_REF = 386

FACE_SIGNATURE_PAIRS = [
    (33, 263), (1, 10), (1, 199), (61, 291), (234, 454),
    (33, 61), (263, 291), (10, 199), (133, 362),
    (55, 285), (105, 334), (1, 152),
]


def face_signature(landmarks, w, h):
    face_l = np.array([landmarks[234].x * w, landmarks[234].y * h])
    face_r = np.array([landmarks[454].x * w, landmarks[454].y * h])
    face_w_norm = np.linalg.norm(face_r - face_l) + 1e-6

    sig = []
    for a, b in FACE_SIGNATURE_PAIRS:
        pa = np.array([landmarks[a].x * w, landmarks[a].y * h])
        pb = np.array([landmarks[b].x * w, landmarks[b].y * h])
        dist = np.linalg.norm(pa - pb) / face_w_norm
        sig.append(dist)
    return np.array(sig)


def signature_distance(sig1, sig2):
    return np.linalg.norm(sig1 - sig2)


def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter + 1e-6
    return inter / union


class FaceDatabase:
    def __init__(self):
        self.people = {}
        self.next_id = 1

    def match_or_create(self, signature, bbox, frame_idx):
        best_id = None
        best_score = float('inf')

        for pid, data in self.people.items():
            sig_dist = signature_distance(signature, data['signature'])
            iou_score = iou(bbox, data['last_bbox']) if data['frames_absent'] < 5 else 0
            combined = sig_dist - 0.05 * iou_score

            if combined < best_score:
                best_score = combined
                best_id = pid

        if best_id is not None and best_score < FACE_MATCH_THRESHOLD:
            person = self.people[best_id]
            person['signature'] = 0.9 * person['signature'] + 0.1 * signature
            person['last_bbox'] = bbox
            person['frames_absent'] = 0
            person['last_seen_frame'] = frame_idx
            person['total_seen'] += 1
            return best_id
        else:
            new_id = self.next_id
            self.next_id += 1
            self.people[new_id] = {
                'signature':    signature.copy(),
                'last_bbox':    bbox,
                'frames_absent': 0,
                'first_seen_frame': frame_idx,
                'last_seen_frame':  frame_idx,
                'total_seen':   1,
            }
            return new_id

    def mark_absent(self, seen_ids):
        for pid, data in self.people.items():
            if pid not in seen_ids:
                data['frames_absent'] += 1


face_db = FaceDatabase()


def head_pose(landmarks, w, h):
    nose_tip  = np.array([landmarks[1].x   * w, landmarks[1].y   * h])
    left_eye  = np.array([landmarks[33].x  * w, landmarks[33].y  * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    chin      = np.array([landmarks[199].x * w, landmarks[199].y * h])
    forehead  = np.array([landmarks[10].x  * w, landmarks[10].y  * h])

    eye_center = (left_eye + right_eye) / 2
    eye_dist   = np.linalg.norm(right_eye - left_eye) + 1e-6
    yaw = (nose_tip[0] - eye_center[0]) / eye_dist * 90

    face_height = np.linalg.norm(chin - forehead) + 1e-6
    nose_ratio  = (nose_tip[1] - forehead[1]) / face_height
    pitch = (nose_ratio - 0.55) * 180
    return pitch, yaw


def ear(landmarks, eye_idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    vert = np.linalg.norm(np.subtract(pts[1], pts[5])) + np.linalg.norm(np.subtract(pts[2], pts[4]))
    horz = np.linalg.norm(np.subtract(pts[0], pts[3]))
    return vert / (2.0 * horz + 1e-6)


def mar(landmarks, w, h):
    top    = np.array([landmarks[13].x  * w, landmarks[13].y  * h])
    bottom = np.array([landmarks[14].x  * w, landmarks[14].y  * h])
    left   = np.array([landmarks[61].x  * w, landmarks[61].y  * h])
    right  = np.array([landmarks[291].x * w, landmarks[291].y * h])
    vert = np.linalg.norm(top - bottom)
    horz = np.linalg.norm(left - right)
    return vert / (horz + 1e-6)


def gaze_ratio(landmarks, iris_center_idx, corners, vert, w, h):
    iris  = np.array([landmarks[iris_center_idx].x * w, landmarks[iris_center_idx].y * h])
    p_out = np.array([landmarks[corners[0]].x * w,      landmarks[corners[0]].y * h])
    p_in  = np.array([landmarks[corners[1]].x * w,      landmarks[corners[1]].y * h])
    p_top = np.array([landmarks[vert[0]].x * w,         landmarks[vert[0]].y * h])
    p_bot = np.array([landmarks[vert[1]].x * w,         landmarks[vert[1]].y * h])

    eye_center_x = (p_out[0] + p_in[0]) / 2
    eye_width    = abs(p_in[0] - p_out[0]) + 1e-6
    gaze_x = (iris[0] - eye_center_x) / (eye_width / 2)

    eye_center_y = (p_top[1] + p_bot[1]) / 2
    eye_height   = abs(p_bot[1] - p_top[1]) + 1e-6
    gaze_y = (iris[1] - eye_center_y) / (eye_height / 2)
    return gaze_x, gaze_y


def get_gaze(landmarks, w, h):
    lx, ly = gaze_ratio(landmarks, LEFT_IRIS_CENTER,  LEFT_EYE_CORNERS,  LEFT_EYE_VERT,  w, h)
    rx, ry = gaze_ratio(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS, RIGHT_EYE_VERT, w, h)
    gaze_x = (-lx + rx) / 2
    gaze_y = (ly + ry) / 2
    return gaze_x, gaze_y


def detect_mood(landmarks, w, h, avg_ear, mouth_open):
    face_left  = np.array([landmarks[234].x * w, landmarks[234].y * h])
    face_right = np.array([landmarks[454].x * w, landmarks[454].y * h])
    face_w = np.linalg.norm(face_right - face_left) + 1e-6

    mouth_l = np.array([landmarks[MOUTH_LEFT].x  * w, landmarks[MOUTH_LEFT].y  * h])
    mouth_r = np.array([landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h])
    upper   = np.array([landmarks[UPPER_LIP_CENTER].x * w, landmarks[UPPER_LIP_CENTER].y * h])
    lower   = np.array([landmarks[LOWER_LIP_CENTER].x * w, landmarks[LOWER_LIP_CENTER].y * h])
    mouth_center_y = (upper[1] + lower[1]) / 2
    mouth_corner_avg_y = (mouth_l[1] + mouth_r[1]) / 2
    smile_score = (mouth_center_y - mouth_corner_avg_y) / face_w

    brow_l = np.array([landmarks[LEFT_BROW_INNER].x  * w, landmarks[LEFT_BROW_INNER].y  * h])
    brow_r = np.array([landmarks[RIGHT_BROW_INNER].x * w, landmarks[RIGHT_BROW_INNER].y * h])
    brow_gap = np.linalg.norm(brow_r - brow_l) / face_w

    brow_outer_l = np.array([landmarks[LEFT_BROW_OUTER].x  * w, landmarks[LEFT_BROW_OUTER].y  * h])
    brow_outer_r = np.array([landmarks[RIGHT_BROW_OUTER].x * w, landmarks[RIGHT_BROW_OUTER].y * h])
    eye_top_l    = np.array([landmarks[LEFT_EYE_TOP_REF].x  * w, landmarks[LEFT_EYE_TOP_REF].y  * h])
    eye_top_r    = np.array([landmarks[RIGHT_EYE_TOP_REF].x * w, landmarks[RIGHT_EYE_TOP_REF].y * h])
    brow_height = ((eye_top_l[1] - brow_outer_l[1]) + (eye_top_r[1] - brow_outer_r[1])) / 2 / face_w

    # Приоритет сверху вниз
    if mouth_open > YAWN_MAR_THRESHOLD:
        mood = ("BORED", "(yawn)", (180, 180, 0))
    elif avg_ear < SLEEPY_EAR_THRESH and smile_score < SMILE_THRESHOLD:
        mood = ("SLEEPY", "(tired)", (130, 100, 200))
    elif brow_height > BROW_RAISE_THRESH and mouth_open > 0.15:
        mood = ("SURPRISED", "(!)", (255, 200, 0))
    elif smile_score > SMILE_THRESHOLD:
        mood = ("ENGAGED", "(:))", (0, 255, 100))
    elif smile_score < FROWN_THRESHOLD and brow_height < 0.10:
        # CONFUSED теперь требует ОБА сигнала: уголки рта вниз И брови опущены
        mood = ("CONFUSED", "(?)", (0, 140, 255))
    elif smile_score < FROWN_THRESHOLD:
        mood = ("UNHAPPY", "(:()", (100, 100, 255))
    else:
        mood = ("NEUTRAL", "(-)", (200, 200, 200))

    return mood[0], mood[1], mood[2], {"smile": smile_score, "brow_gap": brow_gap, "brow_h": brow_height}

def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


history      = defaultdict(lambda: deque(maxlen=30))
mar_history  = defaultdict(lambda: deque(maxlen=20))
mood_history = defaultdict(lambda: deque(maxlen=15))

cap = cv2.VideoCapture(0)
frame_count = 0
phone_boxes = []

while True:
    ok, img = cap.read()
    if not ok:
        break
    h, w, _ = img.shape
    frame_count += 1

    # --- YOLO для телефонов ---
    if frame_count % 2 == 0:
        yolo_res = yolo(img, verbose=False, conf=0.35)[0]
        phone_boxes = []
        for box in yolo_res.boxes:
            if int(box.cls[0]) == PHONE_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                phone_boxes.append((x1, y1, x2, y2))

    for (x1, y1, x2, y2) in phone_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(img, "PHONE", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 165, 255), 2)

    # --- Face Mesh ---
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    distracted_count = 0
    total = 0
    distraction_ratios = []
    mood_counter = defaultdict(int)
    seen_ids_this_frame = set()

    if res.multi_face_landmarks:
        for fl in res.multi_face_landmarks:
            xs = [lm.x * w for lm in fl.landmark]
            ys = [lm.y * h for lm in fl.landmark]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            # Persistent ID
            sig = face_signature(fl.landmark, w, h)
            person_id = face_db.match_or_create(sig, (x1, y1, x2, y2), frame_count)
            seen_ids_this_frame.add(person_id)
            person_data = face_db.people[person_id]

            pitch, yaw     = head_pose(fl.landmark, w, h)
            gaze_x, gaze_y = get_gaze(fl.landmark, w, h)
            avg_ear        = (ear(fl.landmark, LEFT_EYE, w, h) + ear(fl.landmark, RIGHT_EYE, w, h)) / 2
            mouth_open     = mar(fl.landmark, w, h)

            mood_label, mood_emoji, mood_color, mood_signals = detect_mood(
                fl.landmark, w, h, avg_ear, mouth_open
            )
            mood_history[person_id].append(mood_label)
            mood_counts = {}
            for m in mood_history[person_id]:
                mood_counts[m] = mood_counts.get(m, 0) + 1
            smoothed_mood = max(mood_counts, key=mood_counts.get)

            body_zone = (max(0, x1 - 60), y1, min(w, x2 + 60), min(h, y2 + 250))

            head_sideways = abs(yaw) > YAW_THRESHOLD
            head_down     = pitch > PITCH_DOWN_THRESHOLD
            head_up       = pitch < PITCH_UP_THRESHOLD
            gaze_sideways = abs(gaze_x) > GAZE_X_THRESHOLD
            gaze_down     = gaze_y > GAZE_Y_THRESHOLD
            gaze_up       = gaze_y < -GAZE_Y_THRESHOLD

            looking_away = False
            if gaze_sideways or gaze_down or gaze_up:
                looking_away = True
            elif (head_sideways or head_down or head_up) and not (abs(gaze_x) < 0.2 and abs(gaze_y) < 0.25):
                looking_away = True

            eyes_closed = avg_ear < EAR_THRESHOLD
            phone_near  = any(boxes_overlap(body_zone, pb) for pb in phone_boxes)

            mar_history[person_id].append(mouth_open)
            talking = False
            if len(mar_history[person_id]) >= 10:
                arr = np.array(mar_history[person_id])
                talking = arr.std() > MAR_STD_THRESHOLD and arr.max() > MAR_MAX_THRESHOLD

            mood_bad = smoothed_mood in ("BORED", "SLEEPY")

            bad_frame = looking_away or eyes_closed or phone_near or talking or mood_bad
            history[person_id].append(bad_frame)
            distraction_ratio = sum(history[person_id]) / len(history[person_id])
            distracted = distraction_ratio > DISTRACTION_RATIO
            distraction_ratios.append(distraction_ratio)
            mood_counter[smoothed_mood] += 1

            # --- Опорные точки (жёлтые кружки) ---
            for idx in [33, 263, 1, 61, 291, 199, 10]:
                lm = fl.landmark[idx]
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 255), -1)

            # Стрелка направления головы (фиолетовая)
            nose = fl.landmark[1]
            nx, ny = int(nose.x * w), int(nose.y * h)
            cv2.arrowedLine(img, (nx, ny),
                            (int(nx + yaw * 4), int(ny + pitch * 4)),
                            (255, 0, 255), 3, tipLength=0.3)

            # Стрелки взгляда (жёлтые)
            for iris_idx in (LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
                iris = fl.landmark[iris_idx]
                ix, iy = int(iris.x * w), int(iris.y * h)
                cv2.arrowedLine(img, (ix, iy),
                                (int(ix + gaze_x * 60), int(iy + gaze_y * 40)),
                                (0, 255, 255), 2, tipLength=0.4)

            # Причины
            reasons = []
            if phone_near:                          reasons.append("PHONE")
            if talking:                             reasons.append("TALKING")
            if gaze_sideways:                       reasons.append("EYES OFF")
            if gaze_down:                           reasons.append("EYES DOWN")
            if gaze_up:                             reasons.append("EYES UP")
            if head_sideways and not gaze_sideways: reasons.append("HEAD TURN")
            if head_down and not gaze_down:         reasons.append("HEAD DOWN")
            if eyes_closed:                         reasons.append("EYES CLOSED")
            if mood_bad:                            reasons.append(smoothed_mood)
            reason_text = ", ".join(reasons) if reasons else "FOCUSED"

            color = (0, 0, 255) if distracted else (0, 200, 0)
            seconds_seen = person_data['total_seen'] / 30
            status = reason_text if distracted else "FOCUSED"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"Person #{person_id}", (x1, y1 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(img, f"{smoothed_mood} {mood_emoji}", (x1, y1 - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mood_color, 2)
            cv2.putText(img, status, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            debug_lines = [
                f"yaw:{yaw:+5.1f}  pitch:{pitch:+5.1f}",
                f"gaze x:{gaze_x:+.2f} y:{gaze_y:+.2f}",
                f"ear:{avg_ear:.2f}  mar:{mouth_open:.2f}",
                f"seen:{seconds_seen:.1f}s  dist:{int(distraction_ratio*100)}%"
            ]
            for k, line in enumerate(debug_lines):
                cv2.putText(img, line, (x1, y2 + 18 + k * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)

            total += 1
            if distracted:
                distracted_count += 1

    face_db.mark_absent(seen_ids_this_frame)

    # --- HUD ---
    if distraction_ratios:
        avg_distraction = np.mean(distraction_ratios)
        focus_pct = int(100 * (1 - avg_distraction))
    else:
        focus_pct = 0

    if mood_counter:
        dominant_mood = max(mood_counter, key=mood_counter.get)
    else:
        dominant_mood = "-"

    total_registered = len(face_db.people)
    currently_visible = len(seen_ids_this_frame)

    cv2.rectangle(img, (0, 0), (560, 140), (0, 0, 0), -1)
    cv2.putText(img, f"Focus: {total-distracted_count}/{total}  ({focus_pct}%)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Visible now: {currently_visible}   Registered total: {total_registered}",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
    cv2.putText(img, f"Phones: {len(phone_boxes)}",
                (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    cv2.putText(img, f"Class mood: {dominant_mood}",
                (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)
    cv2.putText(img, "ESC to quit",
                (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.imshow("Classroom Attention", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()