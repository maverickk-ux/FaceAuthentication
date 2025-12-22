import cv2
import pickle
import numpy as np
from numpy.linalg import norm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
from collections import deque

# ================= CONFIG =================
FACE_MODEL = "ArcFace"
DISTANCE_THRESHOLD = 0.55

EMBEDDINGS_PATH = "embeddings/face_embeddings.pkl"

# Emotion config - BALANCED
EMOTION_INTERVAL = 3       # run emotion detection every 3 frames
EMOTION_HISTORY_LEN = 4    # keep last 4 readings
EMOTION_CONFIDENCE_THRESHOLD = 35  # minimum confidence to switch emotions
SCORE_SMOOTHING_ALPHA = 0.3  # exponential smoothing factor (0-1, lower = smoother)

# =========================================

print("[INFO] Loading YOLOv8 model...")
yolo = YOLO("yolov8n.pt")

print("[INFO] Initializing DeepSORT tracker...")
tracker = DeepSort(max_age=30)

print("[INFO] Loading trained face embeddings...")
with open(EMBEDDINGS_PATH, "rb") as f:
    face_db = pickle.load(f)

print(f"[INFO] Loaded identities: {list(face_db.keys())}")

# ================= STATE =================
name_to_pid = {}          # name -> permanent ID
next_person_id = 1

track_identities = {}     # track_id -> (name, confidence)
track_emotions = {}       # track_id -> emotion data

frame_count = 0

# ================= HELPERS =================
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def recognize_face(face_img):
    """
    Recognize face using ArcFace embeddings
    """
    rep = DeepFace.represent(
        img_path=face_img,
        model_name=FACE_MODEL,
        enforce_detection=False
    )[0]["embedding"]

    emb = np.array(rep)
    emb = emb / norm(emb)

    best_name = "Unknown"
    best_score = 0.0

    for name, embeddings in face_db.items():
        scores = [
            1 - cosine_distance(emb, e)
            for e in embeddings
        ]
        avg_score = sum(scores) / len(scores)

        if avg_score > best_score and avg_score > DISTANCE_THRESHOLD:
            best_score = avg_score
            best_name = name

    return best_name, best_score

def detect_emotion_with_scores(face_img):
    """
    Detect emotion with confidence scores
    """
    result = DeepFace.analyze(
        img_path=face_img,
        actions=["emotion"],
        enforce_detection=False
    )

    if isinstance(result, list):
        result = result[0]

    emotion_scores = result["emotion"]
    dominant_emotion = result["dominant_emotion"]
    
    return dominant_emotion, emotion_scores

def smooth_scores(prev_scores, new_scores, alpha):
    """
    Exponential smoothing for emotion scores to reduce variance
    """
    if not prev_scores:
        return new_scores
    
    smoothed = {}
    for emotion in new_scores.keys():
        prev_val = prev_scores.get(emotion, new_scores[emotion])
        smoothed[emotion] = alpha * new_scores[emotion] + (1 - alpha) * prev_val
    
    return smoothed

def get_stable_emotion(history, current_emotion, current_scores, prev_emotion):
    """
    Stable emotion detection with hysteresis to prevent flickering
    """
    # Check if current emotion has strong confidence
    current_confidence = current_scores[current_emotion]
    
    # If we already have a previous emotion, require higher confidence to switch
    if prev_emotion and prev_emotion != current_emotion:
        # Hysteresis: new emotion must be significantly stronger
        prev_confidence = current_scores.get(prev_emotion, 0)
        
        # Only switch if new emotion is at least 10% stronger OR very confident
        if current_confidence < prev_confidence + 10 and current_confidence < EMOTION_CONFIDENCE_THRESHOLD:
            return prev_emotion
    else:
        # Same emotion or first detection - just check threshold
        if current_confidence < EMOTION_CONFIDENCE_THRESHOLD:
            if prev_emotion:
                return prev_emotion
            return current_emotion
    
    # Add current emotion to history
    history.append(current_emotion)
    
    # Count occurrences with recency bias
    emotion_count = {}
    for i, emotion in enumerate(history):
        weight = i + 1  # More recent = higher weight
        if emotion not in emotion_count:
            emotion_count[emotion] = 0
        emotion_count[emotion] += weight
    
    # Return emotion with highest weighted count
    return max(emotion_count, key=emotion_count.get)

# ================= VIDEO =================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Could not open webcam")
    exit()

print("[INFO] Starting real-time face recognition + emotion detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

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

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # -------- Recognition (once per track) --------
        if tid not in track_identities:
            try:
                name, conf_score = recognize_face(face_crop)

                if name != "Unknown":
                    if name not in name_to_pid:
                        name_to_pid[name] = next_person_id
                        next_person_id += 1

                track_identities[tid] = (name, conf_score)
            except:
                track_identities[tid] = ("Unknown", 0.0)

        name, conf_score = track_identities[tid]

        # -------- Emotion Tracking - STABILIZED --------
        if tid not in track_emotions:
            track_emotions[tid] = {
                "history": deque(maxlen=EMOTION_HISTORY_LEN),
                "final": "neutral",
                "raw_scores": {},
                "smoothed_scores": {},
                "display_confidence": 0
            }

        # Run emotion detection
        if frame_count % EMOTION_INTERVAL == 0:
            try:
                # Resize face for faster processing
                face_resized = cv2.resize(face_crop, (224, 224))
                
                emotion, emotion_scores = detect_emotion_with_scores(face_resized)
                
                # Smooth the scores to reduce variance
                prev_smoothed = track_emotions[tid]["smoothed_scores"]
                smoothed_scores = smooth_scores(prev_smoothed, emotion_scores, SCORE_SMOOTHING_ALPHA)
                
                # Get dominant emotion from smoothed scores
                smoothed_dominant = max(smoothed_scores, key=smoothed_scores.get)
                
                # Update with stability checking
                history = track_emotions[tid]["history"]
                prev_emotion = track_emotions[tid]["final"]
                
                stable_emotion = get_stable_emotion(
                    history, 
                    smoothed_dominant, 
                    smoothed_scores,
                    prev_emotion
                )
                
                track_emotions[tid]["final"] = stable_emotion
                track_emotions[tid]["raw_scores"] = emotion_scores
                track_emotions[tid]["smoothed_scores"] = smoothed_scores
                
                # Smooth the display confidence too
                new_confidence = smoothed_scores[stable_emotion]
                old_display = track_emotions[tid]["display_confidence"]
                track_emotions[tid]["display_confidence"] = (
                    0.7 * new_confidence + 0.3 * old_display if old_display else new_confidence
                )
                
            except Exception as e:
                # Silently handle errors but maintain previous emotion
                pass

        final_emotion = track_emotions[tid]["final"]
        confidence = track_emotions[tid]["display_confidence"]

        # -------- Display --------
        if name != "Unknown":
            pid = name_to_pid[name]
            label = f"ID:{pid} {name} | {final_emotion} ({confidence:.0f}%)"
            color = (0, 255, 0)
        else:
            label = f"Unknown | {final_emotion} ({confidence:.0f}%)"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Face Recognition + Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
