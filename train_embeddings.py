import os
import pickle
import numpy as np
from deepface import DeepFace

# ---------------- CONFIG ----------------
FACE_MODEL = "ArcFace"
DATASET_DIR = "dataset"
OUTPUT_PATH = "embeddings/face_embeddings.pkl"
MIN_IMAGES = 5

os.makedirs("embeddings", exist_ok=True)

database = {}

print("[INFO] Training embeddings (CACHE-SAFE VERSION)...")

for person in sorted(os.listdir(DATASET_DIR)):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []
    print(f"\n[INFO] Processing {person}")

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=FACE_MODEL,
                detector_backend="retinaface",
                enforce_detection=True,
                align=True
            )

            embeddings.append(np.array(rep[0]["embedding"]))
            print(f"[OK] {img_name}")

        except Exception as e:
            print(f"[SKIP] {img_name} ({str(e)[:50]})")

    if len(embeddings) >= MIN_IMAGES:
        database[person] = embeddings
        print(f"[DONE] {person}: {len(embeddings)} embeddings saved")
    else:
        print(f"[REJECTED] {person}: only {len(embeddings)} valid faces")

# ---------------- SAVE ----------------
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(database, f)

print("\n[SUCCESS] Training complete")
print("[INFO] Identities:", list(database.keys()))
