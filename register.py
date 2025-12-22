import cv2
import os
import time
import subprocess
import sys

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
IMAGES_REQUIRED = 15
CAPTURE_DELAY = 0.7


def register():
    print("\n========== FACE REGISTRATION ==========\n")
    print("[INFO] Using Python:", sys.executable)

    name = input("Enter Person Name: ").strip()

    if not name:
        print("[ERROR] Name cannot be empty")
        return

    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam")
        return

    print("\n[INFO] Capturing face images")
    print("➡️ Look straight, then slowly turn left / right / up / down\n")

    count = 0
    last_capture = time.time()

    while count < IMAGES_REQUIRED:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        cv2.putText(
            display,
            f"Images Captured: {count}/{IMAGES_REQUIRED}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Register Face | Press Q to Quit", display)

        if time.time() - last_capture >= CAPTURE_DELAY:
            img_path = os.path.join(person_dir, f"img_{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[SAVED] {img_path}")
            count += 1
            last_capture = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count < IMAGES_REQUIRED:
        print("\n[ERROR] Registration incomplete")
        return

    print("\n[INFO] Training embeddings using DeepFace ArcFace...\n")

    result = subprocess.run(
        [sys.executable, "train_embeddings.py"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    if result.returncode != 0:
        print("\n[ERROR] Embedding training failed")
        return

    print("\n[SUCCESS] Registration completed")
    print(f"User: {name}")
    print(f"Images: {count}")
    print("Embeddings updated at embeddings/face_embeddings.pkl\n")


if __name__ == "__main__":
    register()
