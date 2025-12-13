import os
import pickle
import numpy as np
from deepface import DeepFace

FACE_MODEL = "ArcFace"
DATASET_DIR = "dataset"
OUTPUT_PATH = "embeddings/face_embeddings.pkl"

database = {}

print("[INFO] Starting training...")

for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for img in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img)

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=FACE_MODEL,
                enforce_detection=False
            )
            embeddings.append(np.array(rep[0]["embedding"]))
        except Exception as e:
            print(f"[WARN] {person}/{img} skipped")

    if len(embeddings) >= 5:
        database[person] = embeddings
        print(f"[OK] {person}: {len(embeddings)} images trained")
    else:
        print(f"[SKIP] {person}: not enough images")

os.makedirs("embeddings", exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(database, f)

print(f"[DONE] Training completed. Saved to {OUTPUT_PATH}")
