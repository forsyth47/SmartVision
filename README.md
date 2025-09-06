# Professional CCTV Face Recognition System

## Overview

This project implements a professional-grade face recognition system for CCTV and video analytics, inspired by industry best practices (SCRFD+ArcFace, FaceRec). It is designed to prevent false positives and deliver accurate, explainable results using:
- L2-normalized face embeddings
- Cosine similarity with strict, configurable thresholds
- Quality and demographic-aware filtering
- A modern Streamlit web interface for search and analysis

---

## Features
- **Professional Ingestion Pipeline**: Extracts faces from video, applies quality control, and stores L2-normalized embeddings in a Qdrant vector database.
- **Accurate Face Search**: Uses strict similarity thresholds and demographic filtering to prevent false matches.
- **Explainable Results**: Shows similarity scores, demographics, and quality metrics for each match.
- **Modern Web UI**: Streamlit app for uploading query images, configuring thresholds, and visualizing results.
- **Debugging Tools**: See near-miss results and similarity distributions for transparency.

---

## Project Structure

```
/Users/zero/Documents/Code/Vsolv/PythonProject/
├── main_ingest.py            # Professional ingestion (database creation)
├── main_app.py               # Professional Streamlit search app
├── face_recognition_engine.py# Core recognition engine (L2, cosine, thresholds)
├── requirements.txt          # All dependencies
├── sample.mp4                # Example video for ingestion
├── data/
│   ├── faces/                # Cropped face images
│   ├── preview_faces/        # Preview images for UI
│   └── metadata.json         # Metadata for all faces
├── qdrant_db/                # Vector database files
└── ...
```

---

## Installation

1. **Clone the repository**

```bash
cd SmartVision
```

2. **Install Python 3.10+ and [pip](https://pip.pypa.io/en/stable/)**

3. **Create a virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Ingest Faces from Video

This step processes a video (e.g., `sample.mp4`), detects faces, applies quality control, and builds the vector database.

```bash
python main_ingest.py
```
- Output: L2-normalized embeddings, face crops, and metadata in `data/` and `qdrant_db/`.
- You can replace `sample.mp4` with your own video (ensure it's in the project root).

### 2. Run the Web Search App

Start the Streamlit app for searching faces:

```bash
streamlit run main_app.py
```
- Open the provided local URL (e.g., http://localhost:8501) in your browser.
- Upload the provided sample face image to search for matches in the database. (Provided in the folder)
- Adjust similarity thresholds in the sidebar for stricter or looser matching.

---

## How It Works (Detailed)

### Ingestion Pipeline (`main_ingest.py`)
1. **Frame Extraction**: Samples frames from the video at regular intervals.
2. **Face Detection**: Uses multiple backends (RetinaFace, MTCNN, OpenCV) for robust detection.
3. **Quality Assessment**: Each face is scored for sharpness, contrast, brightness, size, and structure. Only high-quality faces are kept.
4. **Embedding Generation**: Each face is encoded using ArcFace, then L2-normalized for cosine similarity.
5. **Demographic Analysis**: Gender, age range, emotion, and race are estimated for each face.
6. **Database Storage**: Embeddings and metadata are stored in Qdrant (cosine distance). Cropped faces and previews are saved in `data/`.

### Search & Recognition (`main_app.py`)
1. **Image Upload**: User uploads a query image via the web UI.
2. **Embedding & Demographics**: The system generates a normalized embedding and analyzes demographics for the query face.
3. **Similarity Search**: The embedding is compared to the database using cosine similarity. Only results above the selected threshold are shown.
4. **Demographic Filtering**: Matches are boosted/penalized based on gender and age similarity.
5. **Result Display**: Each match shows the face crop, similarity score, demographics, and technical details. Images are shown with an invert and 90° rotation filter for visual distinction.
6. **Debugging**: If no matches are found, the closest (below-threshold) results are shown for transparency.

### Thresholds & Quality
- **Strict**: 0.65 (very high confidence, few false positives)
- **Normal**: 0.55 (recommended for most use cases)
- **Loose**: 0.45 (higher recall, more false positives)
- **Minimum**: 0.35 (very permissive, use with caution)
- **Quality Filtering**: Faces with quality <0.4 are excluded from the database.

---

## Best Practices & Notes
- For best results, use high-quality, well-lit videos for ingestion.
- Always re-run ingestion if you change the video or want to update the database.
- The system is designed to **prevent false matches**—random faces should not match unless truly similar.
- All face crops and metadata are stored for auditability and debugging.

---

## Troubleshooting
- **No faces found in database?**
  - Make sure you ran `python main_ingest.py` and have a valid video file.
- **App won't start?**
  - Check Python version (3.10+), virtual environment, and that all dependencies are installed.
- **Face not detected in query?**
  - Use a clear, front-facing image with a single face.
- **Performance issues?**
  - For large videos, ingestion may take several minutes.

---

## Dependencies
See `requirements.txt` for all dependencies. Key packages:
- `deepface` (face detection, embedding, demographics)
- `opencv-python` (image processing)
- `qdrant-client` (vector database)
- `streamlit` (web UI)
- `numpy`, `pillow`, `tensorflow`

---

## Credits
- Inspired by SCRFD+ArcFace, FaceRec, and best practices from commercial face recognition systems.
- Built with open-source tools for research and educational use.

---

## License
This project is for research and educational purposes only. Commercial use may require additional licensing for some dependencies (e.g., ArcFace, DeepFace).
