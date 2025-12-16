import cv2
import pickle
import numpy as np
from numpy.linalg import norm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
from collections import defaultdict

# ---------------- CONFIG ----------------
FACE_MODEL = "ArcFace"
DISTANCE_THRESHOLD = 0.50
VOTE_THRESHOLD = 3
EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"

# ---------------- LOAD MODELS ----------------
print("[INFO] Loading YOLOv8 model...")
yolo = YOLO("yolov8n.pt")

print("[INFO] Initializing DeepSORT tracker...")
tracker = DeepSort(max_age=30)

print("[INFO] Loading trained face embeddings...")
with open(EMBEDDINGS_PATH, "rb") as f:
    face_db = pickle.load(f)

print(f"[INFO] Loaded identities: {list(face_db.keys())}")

# ---------------- ID MANAGEMENT ----------------
name_to_pid = {}
next_person_id = 1

# Per-track temporal memory
track_memory = defaultdict(lambda: {
    "votes": defaultdict(int),
    "locked_name": None,
    "confidence": 0.0
})

# ---------------- HELPER FUNCTIONS ----------------
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def recognize_face_trained(face_img):
    reps = DeepFace.represent(
        img_path=face_img,
        model_name=FACE_MODEL,
        detector_backend="retinaface",
        enforce_detection=True
    )

    if len(reps) == 0:
        return "Unknown", 0.0

    emb = np.array(reps[0]["embedding"])

    best_name = "Unknown"
    best_score = 0.0

    for name, embeddings in face_db.items():
        scores = [
            1 - cosine_distance(emb, e)
            for e in embeddings
        ]

        person_best = max(scores)

        if person_best > best_score and person_best > DISTANCE_THRESHOLD:
            best_score = person_best
            best_name = name

    return best_name, best_score

# ---------------- VIDEO CAPTURE ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open camera")
    exit()

print("[INFO] Starting real-time face recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- Person Detection --------
    results = yolo.predict(frame, classes=[0], verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf > 0.6:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

    # -------- Tracking --------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        mem = track_memory[tid]

        # -------- Recognition with temporal voting --------
        if mem["locked_name"] is None:
            h = y2 - y1
            face_crop = frame[y1 : y1 + int(0.6 * h), x1:x2]

            try:
                name, conf_score = recognize_face_trained(face_crop)

                if name != "Unknown":
                    mem["votes"][name] += 1

                    if mem["votes"][name] >= VOTE_THRESHOLD:
                        mem["locked_name"] = name
                        mem["confidence"] = conf_score

                        if name not in name_to_pid:
                            name_to_pid[name] = next_person_id
                            next_person_id += 1
            except Exception:
                pass

        # -------- Display --------
        if mem["locked_name"] is not None:
            name = mem["locked_name"]
            conf_score = mem["confidence"]
            pid = name_to_pid[name]
            label = f"ID:{pid} {name} ({int(conf_score * 100)}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Face Tracking & Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
