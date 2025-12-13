# AI-Based Face Recognition & Authentication System

This project implements a **robust, real-time face recognition system** using state-of-the-art deep learning models. The system is designed to be **accurate, efficient, and scalable**, and serves as the foundation for a **Face Authentication Website**.

The solution follows **industry-standard practices** by using deep face embeddings instead of training a neural network from scratch, ensuring high accuracy across **different face angles, lighting conditions, and real-world scenarios**.

---

## Key Features

### Real-Time Face Detection & Tracking
- **YOLOv8** for real-time person detection from live webcam feed.
- **DeepSORT** for reliable multi-object tracking across frames.
- Smooth tracking even when people leave and re-enter the frame.

---

### High-Accuracy Face Recognition (ArcFace)
- Uses **ArcFace**, a state-of-the-art face recognition model.
- Robust against:
  - Pose and angle variations
  - Lighting changes
  - Minor occlusions (glasses, facial hair)

---

### AI Training Using Multi-Image Identity Templates
Instead of retraining a CNN, the system uses **embedding-based identity learning**:

- Each person is trained using **10–20 images** captured from different angles and lighting.
- Face embeddings are extracted and stored as identity templates.
- Recognition is performed by comparing live embeddings against multiple stored embeddings per person.

This approach significantly improves accuracy and stability.

---

### Permanent Identity Assignment
- Each recognized person is assigned a **permanent ID**.
- The same ID is preserved even if the person leaves and re-enters the frame.
- Prevents identity duplication and redundancy.

**Example:**  ID:1 Shashank

---

### Confidence-Based Recognition
- Displays a **confidence score (%)** for each recognized face.
- Confidence is computed using **cosine similarity** across multiple embeddings.
- Helps make reliable authentication decisions.

**Example:** ID:1 Shashank (94%)


---

### Optimized & CPU-Friendly Design
- No redundant model loading during runtime.
- Face recognition is executed **once per tracked person**, not every frame.
- Fully functional on **CPU-only systems** (no GPU required).

---

## Technology Stack

- **Python**
- **YOLOv8** – Person Detection
- **DeepSORT** – Object Tracking
- **DeepFace (ArcFace)** – Face Recognition
- **OpenCV** – Video Processing
- **NumPy / Pickle** – Embedding Storage & Similarity Computation

---

## Project Structure
```md
Face_Recognition/
├── dataset/                       # Training images (per person)
│ └── Person_Name/
│ ├── img1.jpg
│ ├── img2.jpg
│ └── ...
├── embeddings/                    # Trained face embeddings
│ └── face_embeddings.pkl
├── train_embeddings.py            # AI training script
├── track.py                       # Real-time recognition & tracking
├── yolov8n.pt                     # YOLOv8 model
├── venv/                          # Virtual environment
```

---

## Setup Instructions

###  Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
### Install Dependencies
```bash
pip install opencv-python deepface ultralytics deep-sort-realtime tf-keras numpy
```

## Training the Face Recognition Model

- 1. Add 10–20 images per person inside the dataset/ folder.
- 2. Run the training script:

```bash
python train_embeddings.py
```

- This generates:
```bash
embeddings/face_embeddings.pkl
```
- This acts like a trained AI identity database
---

## Running Real-Time Face Recognition
```bash
python track.py
```
- Opens webcam
- Detects, tracks, and recognizes faces
- Displays permanent ID, name, and confidence score
- Press ```q``` to exit

---

## Future Enchancements (To be Updated)
- Face Authentication API using FastAPI
- Secure login via multi-frame verification
- Web-based UI for:
  - Face login
  - User registration
- Attendance and access-control applications
- Liveness detection for spoof prevention

---

## Project Members
- K V Shashank Pai
- Tejas Kollipara
- Varun E


