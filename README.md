# AI-Based Face Recognition & Emotion Detection System

This project implements a **robust, real-time face recognition and emotion detection system** using state-of-the-art deep learning models. The system is designed to be **accurate, efficient, and scalable**, serving as the foundation for a **Face Authentication & Emotion Analysis Platform**.

The solution follows **industry-standard practices** by using deep face embeddings instead of training a neural network from scratch, ensuring high accuracy across **different face angles, lighting conditions, and real-world scenarios**.

---

## Key Features

### Real-Time Face Detection & Tracking
- **YOLOv8** for real-time person detection from live webcam feed
- **DeepSORT** for reliable multi-object tracking across frames
- Smooth tracking even when people leave and re-enter the frame

---

### High-Accuracy Face Recognition (ArcFace)
- Uses **ArcFace**, a state-of-the-art face recognition model
- Robust against:
  - Pose and angle variations
  - Lighting changes
  - Minor occlusions (glasses, facial hair)

---

### Real-Time Emotion Detection
- **Integrated emotion analysis** using DeepFace's emotion recognition module
- Detects 7 core emotions: happy, sad, angry, surprise, fear, disgust, neutral
- **Stabilized emotion tracking** with:
  - Exponential smoothing to reduce score variance
  - Hysteresis mechanism to prevent emotion flickering
  - Confidence-based emotion switching (requires 35%+ confidence)
  - Historical emotion tracking with recency bias

**Example:** `ID:1 Shashank | happy (87%)`

---

### AI Training Using Multi-Image Identity Templates
Instead of retraining a CNN, the system uses **embedding-based identity learning**:

- Each person is trained using **10–20 images** captured from different angles and lighting
- Face embeddings are extracted and stored as identity templates
- Recognition is performed by comparing live embeddings against multiple stored embeddings per person

This approach significantly improves accuracy and stability.

---

### Permanent Identity Assignment
- Each recognized person is assigned a **permanent ID**
- The same ID is preserved even if the person leaves and re-enters the frame
- Prevents identity duplication and redundancy

**Example:** `ID:1 Shashank | happy (87%)`

---

### Confidence-Based Recognition
- Displays a **confidence score (%)** for both face recognition and emotion detection
- Face confidence computed using **cosine similarity** across multiple embeddings
- Emotion confidence based on smoothed probability scores
- Helps make reliable authentication and emotion analysis decisions

---

### Optimized & CPU-Friendly Design
- No redundant model loading during runtime
- Face recognition executed **once per tracked person**, not every frame
- Emotion detection runs every 3 frames (configurable) for performance
- Fully functional on **CPU-only systems** (no GPU required)

---

## Technology Stack

- **Python**
- **YOLOv8** – Person Detection
- **DeepSORT** – Multi-Object Tracking
- **DeepFace (ArcFace)** – Face Recognition & Emotion Detection
- **OpenCV** – Video Processing & Display
- **NumPy / Pickle** – Embedding Storage & Similarity Computation

---

## Project Structure
```
Face_Recognition/
├── dataset/                       # Training images (per person)
│   └── Person_Name/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── embeddings/                    # Trained face embeddings
│   └── face_embeddings.pkl
├── train_embeddings.py            # AI training script
├── track.py                       # Real-time recognition & emotion detection
├── yolov8n.pt                     # YOLOv8 model
├── venv/                          # Virtual environment
└── README.md
```

---

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

### 2. Install Dependencies
```bash
pip install opencv-python deepface ultralytics deep-sort-realtime tf-keras numpy
```

---

## Training the Face Recognition Model

### 1. User Registration
- Run the registration script:
```
python register.py
```
- What happens automatically:
  - Prompts for the person’s name
  - Opens the webcam
  - Captures ~15 face images at different angles
  - Stores images in:
    ``` dataset/<Person_Name>/```
  - Automatically runs the training pipeline
  - Generates / updates:
    ``` embeddings/face_embeddings.pkl```
    
- After registration, the dataset will look like this:
  ```
  dataset/
  ├── Shashank/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── Tejas/
  │   └── ...
  └── Varun/
      └── ...
  ```
Each person has a dedicated folder containing multiple face images captured from the webcam.

---

## Running Real-Time Face Recognition & Emotion Detection

```bash
python track.py
```

**What happens:**
- Opens webcam feed
- Detects and tracks people in real-time
- Recognizes faces and displays permanent ID with name
- Analyzes and displays:
  - Person name (if recognized)
  - Permanent ID
  - Emotion label with confidence score
- Press `q` to exit

**Sample Output:**
- Recognized: `ID:1 Shashank | happy (87%)`
- Unknown: `Unknown | neutral (72%)`

---

## Configuration Options

Inside `track.py`, you can adjust:

```python
# Face Recognition
DISTANCE_THRESHOLD = 0.55      # Recognition sensitivity (lower = stricter)

# Emotion Detection Performance
EMOTION_INTERVAL = 3           # Run emotion detection every N frames
EMOTION_HISTORY_LEN = 4        # Number of past emotions to track
EMOTION_CONFIDENCE_THRESHOLD = 35  # Minimum confidence to switch emotions
SCORE_SMOOTHING_ALPHA = 0.3    # Smoothing factor (0-1, lower = smoother)
```

---

## Future Enhancements

- [ ] Face Authentication API using FastAPI
- [ ] Secure login via multi-frame verification
- [ ] Web-based UI for:
  - Face login
  - User registration
  - Real-time emotion dashboard
- [ ] Attendance and access-control applications
- [ ] Liveness detection for spoof prevention
- [ ] Emotion-based analytics and reporting
- [ ] Multi-camera support
- [ ] Cloud deployment

---

## Project Members
- **K V Shashank Pai**
- **Tejas Kollipara**
- **Varun E**

---

## License
This project is for educational purposes. Please ensure compliance with privacy laws when deploying face recognition systems.

---

## Acknowledgments
- YOLOv8 by Ultralytics
- DeepFace by Serengil
- DeepSORT tracking algorithm
- ArcFace face recognition model
