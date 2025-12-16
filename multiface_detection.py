import cv2
import pickle
import numpy as np
from numpy.linalg import norm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
from collections import defaultdict

# ================= CONFIG =================
FACE_MODEL = "ArcFace"

# Tuned for LIVE webcam
DISTANCE_THRESHOLD = 0.50
MARGIN = 0.08

VOTE_THRESHOLD = 3
RECOGNIZE_EVERY_N_FRAMES = 3

EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"

# ================= LOAD MODELS =================
print("[INFO] Loading YOLOv8...")
yolo = YOLO("yolov8n.pt")

print("[INFO] Initializing DeepSORT...")
tracker = DeepSort(max_age=30)

print("[INFO] Loading face embeddings...")
with open(EMBEDDINGS_PATH, "rb") as f:
    face_db = pickle.load(f)

print("[INFO] Loaded identities:", list(face_db.keys()))

# ================= STATE =================
name_to_pid = {}
next_person_id = 1
frame_count = 0

track_memory = defaultdict(lambda: {
    "votes": defaultdict(float),
    "locked_name": None,
    "confidence": 0.0
})

# ================= HELPERS =================
def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def recognize_face(person_crop):
    """
    IMPORTANT:
    Uses SAME pipeline as training:
    DeepFace.represent + retinaface + align
    """
    reps = DeepFace.represent(
        img_path=person_crop,
        model_name=FACE_MODEL,
        detector_backend="retinaface",
        enforce_detection=False,
        align=True
    )

    if not reps:
        return "Unknown", 0.0

    emb = np.array(reps[0]["embedding"])

    scores = []
    for name, embeddings in face_db.items():
        sims = [cosine_sim(emb, e) for e in embeddings]
        scores.append((name, max(sims)))

    scores.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0

    print(f"[DEBUG] {best_name}: {best_score:.3f} | second: {second_score:.3f}")

    if best_score < DISTANCE_THRESHOLD:
        return "Unknown", 0.0

    if (best_score - second_score) < MARGIN:
        return "Unknown", 0.0

    return best_name, best_score

# ================= VIDEO =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("[INFO] Running FINAL multi-face recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    H, W, _ = frame.shape

    # -------- PERSON DETECTION --------
    results = yolo.predict(frame, classes=[0], verbose=False)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf > 0.6:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

    tracks = tracker.update_tracks(detections, frame=frame)

    # Prevent same identity stealing
    locked_names = {
        mem["locked_name"]
        for mem in track_memory.values()
        if mem["locked_name"] is not None
    }

    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        mem = track_memory[tid]

        # -------- RECOGNITION --------
        if mem["locked_name"] is None and frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            name, score = recognize_face(person_crop)

            if name in locked_names:
                name = "Unknown"

            if name != "Unknown":
                mem["votes"][name] += 1.0

                if mem["votes"][name] >= VOTE_THRESHOLD and score > 0.55:
                    mem["locked_name"] = name
                    mem["confidence"] = score

                    if name not in name_to_pid:
                        name_to_pid[name] = next_person_id
                        next_person_id += 1

            # Vote decay (prevents early wrong lock)
            for k in list(mem["votes"].keys()):
                mem["votes"][k] *= 0.85

        # -------- DRAW --------
        if mem["locked_name"]:
            pid = name_to_pid[mem["locked_name"]]
            label = f"ID:{pid} {mem['locked_name']} ({int(mem['confidence'] * 100)}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    cv2.imshow("FINAL Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
