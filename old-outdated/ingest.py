import cv2
import os
import json
import uuid
import numpy as np

# Fix TensorFlow issues before importing DeepFace
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    # For compatibility with different TF versions
    if not hasattr(tf, '__version__'):
        tf.__version__ = '2.13.0'
except ImportError:
    print("TensorFlow not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0"])
    import tensorflow as tf

from deepface import DeepFace
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceIngestor:
    def __init__(self):
        """Initialize the face ingestor with Qdrant client and create collection."""
        # Initialize Qdrant client with persistent storage
        self.client = QdrantClient(path="./qdrant_db")  # Persistent database
        self.collection_name = "faces"
        self.vector_size = 512  # ArcFace produces 512-dimensional vectors

        # CRITICAL: Clear existing collection to avoid bad data accumulation
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(self.collection_name)
                logger.info("🗑️ Cleared existing collection to avoid data contamination")
            except Exception:
                logger.info("No existing collection to clear")

            # Create fresh collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            logger.info("✅ Created fresh collection")
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")

        # Create directories
        os.makedirs("data/faces", exist_ok=True)

        self.metadata = []

    def extract_frames(self, video_path: str, frame_interval: int = 150) -> List[tuple]:
        """
        Extract frames from video at specified intervals.

        Args:
            video_path: Path to the video file
            frame_interval: Extract every Nth frame (default: 150 frames ≈ 5 seconds at 30fps)

        Returns:
            List of (frame_number, frame_image) tuples
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames.append((frame_count, frame))
                logger.info(f"Extracted frame {frame_count}")

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames

    def detect_and_embed_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame and generate embeddings using the best available models for CCTV accuracy.

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            List of dictionaries containing face data
        """
        faces_data = []

        try:
            # Convert BGR to RGB for DeepFace
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save debug frame
            debug_frame_path = "debug_frame.jpg"
            cv2.imwrite(debug_frame_path, frame)
            logger.info(f"Frame shape: {frame.shape}, saved debug frame to {debug_frame_path}")

            # Enhanced preprocessing for CCTV footage
            frame_enhanced = self.enhance_frame_for_detection(frame_rgb)

            # Use the most accurate detection and embedding combinations
            detection_configs = [
                # Best accuracy models first (state-of-the-art)
                {'backend': 'retinaface', 'model': 'ArcFace'},
                {'backend': 'mtcnn', 'model': 'ArcFace'},
                {'backend': 'retinaface', 'model': 'Facenet512'},
                {'backend': 'mtcnn', 'model': 'Facenet512'},
                # Fallback for difficult cases
                {'backend': 'opencv', 'model': 'ArcFace'},
                {'backend': 'ssd', 'model': 'ArcFace'},
            ]

            for config in detection_configs:
                try:
                    logger.info(f"Trying {config['backend']} detection with {config['model']} embedding...")

                    # Use enhanced preprocessing and standard parameters
                    representations = DeepFace.represent(
                        img_path=frame_enhanced,
                        model_name=config['model'],
                        detector_backend=config['backend'],
                        enforce_detection=False,  # Changed to False for consistency
                        align=True,  # Enable alignment for better quality
                        normalization='ArcFace'
                    )

                    if representations and len(representations) > 0:
                        logger.info(f"✅ {config['backend']} found {len(representations)} faces with {config['model']}")

                        for i, representation in enumerate(representations):
                            embedding_vector = representation['embedding']
                            face_area = representation.get('facial_area', {})

                            # Enhanced face extraction with quality checks
                            face_crop = self.extract_high_quality_face(frame_rgb, face_area)

                            if face_crop is not None:
                                # Quality assessment
                                quality_score = self.assess_face_quality(face_crop)
                                logger.info(f"Face {i+1} quality score: {quality_score:.3f}")

                                # BIAS PREVENTION: Apply quality normalization to prevent high-quality faces from dominating
                                # Normalize quality score to prevent extreme bias
                                normalized_quality = self.normalize_quality_for_bias_prevention(quality_score)

                                # DEMOGRAPHIC ANALYSIS: Extract demographic information for better matching
                                demographics = self.analyze_face_demographics(face_crop)
                                logger.info(f"Face {i+1} demographics: {demographics}")

                                # Only process faces with reasonable quality (but not just the highest quality ones)
                                if normalized_quality > 0.25:  # Lower threshold to include more diverse faces
                                    # Debug save
                                    debug_face_path = f"debug_face_{config['backend']}_{i}.jpg"
                                    face_uint8 = (face_crop * 255).astype(np.uint8)
                                    cv2.imwrite(debug_face_path, cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR))

                                    face_data = {
                                        'face_image': face_crop,
                                        'embedding': embedding_vector,
                                        'face_id': str(uuid.uuid4()),
                                        'detection_backend': config['backend'],
                                        'embedding_model': config['model'],
                                        'face_area': face_area,
                                        'quality_score': quality_score,
                                        'normalized_quality': normalized_quality,
                                        **demographics  # Add demographic data
                                    }
                                    faces_data.append(face_data)
                                    logger.info(f"✅ Processed face {i+1} with {config['backend']} + {config['model']} (quality: {quality_score:.3f}, normalized: {normalized_quality:.3f}, gender: {demographics.get('gender', 'unknown')})")
                                else:
                                    logger.warning(f"Skipped face {i+1} due to low normalized quality (original: {quality_score:.3f}, normalized: {normalized_quality:.3f})")

                        # If we found good quality faces, use them
                        if faces_data:
                            break

                except Exception as e:
                    logger.warning(f"❌ {config['backend']} failed: {str(e)}")
                    continue

            # If no high-quality faces found, DON'T use aggressive detection to avoid false positives
            if not faces_data:
                logger.info("No high-quality faces found in this frame - skipping to avoid false positives")
                # faces_data = self.aggressive_face_detection(frame_rgb)  # DISABLED
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        logger.info(f"Total high-quality faces processed: {len(faces_data)}")
        return faces_data

    def enhance_frame_for_detection(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality for better face detection in CCTV footage.

        Args:
            frame_rgb: Input RGB frame

        Returns:
            Enhanced frame
        """
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)

            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge([l, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

            # Apply gentle denoising
            enhanced_rgb = cv2.bilateralFilter(enhanced_rgb, 5, 50, 50)

            return enhanced_rgb

        except Exception as e:
            logger.warning(f"Frame enhancement failed: {e}")
            return frame_rgb

    def get_detection_params(self, backend: str, confidence: float) -> dict:
        """
        Get optimized detection parameters for each backend.

        Args:
            backend: Detection backend name
            confidence: Confidence threshold

        Returns:
            Parameter dictionary
        """
        params = {}

        if backend == 'mtcnn':
            params.update({
                'min_face_size': 15,  # Detect smaller faces
                'scale_factor': 0.709,
                'steps_threshold': [0.6, 0.7, confidence]
            })
        elif backend == 'retinaface':
            params.update({
                'threshold': confidence,
                'allow_upscaling': True
            })
        elif backend == 'opencv':
            params.update({
                'scale_factor': 1.05,
                'min_neighbors': 3,
                'min_size': (20, 20)
            })

        return params

    def extract_high_quality_face(self, frame_rgb: np.ndarray, face_area: dict, padding: float = 0.4) -> np.ndarray:
        """
        Extract face region with enhanced quality processing.

        Args:
            frame_rgb: RGB frame
            face_area: Face coordinates from DeepFace
            padding: Padding around face (increased for better context)

        Returns:
            High-quality face crop
        """
        try:
            if face_area and all(k in face_area for k in ['x', 'y', 'w', 'h']):
                x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']

                # Add generous padding for context
                pad_w = int(w * padding)
                pad_h = int(h * padding)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(frame_rgb.shape[1], x + w + pad_w)
                y2 = min(frame_rgb.shape[0], y + h + pad_h)

                if x2 > x1 and y2 > y1:
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        # Use higher resolution for better embeddings
                        face_crop = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_CUBIC)

                        # Apply enhancement specifically for face region
                        face_crop = self.enhance_face_region(face_crop)

                        return face_crop.astype(np.float32) / 255.0

            # Fallback: use center region with enhancement
            h, w = frame_rgb.shape[:2]
            center_crop = frame_rgb[h//4:3*h//4, w//4:3*w//4]
            center_crop = cv2.resize(center_crop, (256, 256), interpolation=cv2.INTER_CUBIC)
            center_crop = self.enhance_face_region(center_crop)
            return center_crop.astype(np.float32) / 255.0

        except Exception as e:
            logger.warning(f"Error extracting high-quality face: {e}")
            return None

    def enhance_face_region(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Apply specific enhancements to face region for better embedding quality.

        Args:
            face_crop: Face region as numpy array

        Returns:
            Enhanced face region
        """
        try:
            # Ensure we have a valid uint8 image
            if face_crop.dtype != np.uint8:
                face_crop = (face_crop * 255).astype(np.uint8) if face_crop.max() <= 1.0 else face_crop.astype(np.uint8)

            # Ensure we have 3 channels
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                # Convert to YUV for better processing
                yuv = cv2.cvtColor(face_crop, cv2.COLOR_RGB2YUV)

                # Enhance Y channel (luminance) - Y channel is grayscale (CV_8UC1)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])

                # Convert back to RGB
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            else:
                # Fallback: if not 3-channel, use the original
                enhanced = face_crop

            # Apply unsharp masking for better details
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255)

            return enhanced

        except Exception as e:
            logger.warning(f"Face enhancement failed: {e}")
            return face_crop

    def assess_face_quality(self, face_crop: np.ndarray) -> float:
        """
        Assess the quality of a detected face for filtering.

        Args:
            face_crop: Face region as numpy array (0-1 range)

        Returns:
            Quality score (0-1, higher is better)
        """
        try:
            # Convert to grayscale for analysis
            face_uint8 = (face_crop * 255).astype(np.uint8)
            gray = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2GRAY)

            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize to 0-1

            # Calculate contrast
            contrast = gray.std() / 255.0

            # Calculate brightness (avoid too dark/bright faces)
            brightness = gray.mean() / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Penalize extreme brightness

            # Calculate face size score (larger faces are generally better)
            size_score = min(min(face_crop.shape[:2]) / 100.0, 1.0)

            # Combined quality score
            quality = (sharpness_score * 0.4 + contrast * 0.3 + brightness_score * 0.2 + size_score * 0.1)

            return quality

        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default medium quality

    def normalize_quality_for_bias_prevention(self, quality_score: float) -> float:
        """
        Normalize quality score to prevent bias towards high-quality faces.

        Args:
            quality_score: Original quality score (0-1 range)

        Returns:
            Normalized quality score (0-1 range)
        """
        try:
            # Apply a non-linear transformation to reduce the impact of high-quality faces
            normalized_score = quality_score ** 0.5  # Square root transformation

            return normalized_score

        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return quality_score  # Fallback to original score

    def aggressive_face_detection(self, frame_rgb: np.ndarray) -> List[Dict]:
        """
        Aggressive face detection for difficult cases using multiple approaches.
        """
        faces_data = []

        try:
            # Try with enhanced preprocessing
            enhanced_frame = self.enhance_frame_for_detection(frame_rgb)

            # Use multiple detection strategies
            detection_strategies = [
                {'backend': 'opencv', 'scale_factors': [1.05, 1.1, 1.2], 'min_neighbors': [1, 2, 3]},
                {'backend': 'mtcnn', 'min_face_sizes': [10, 15, 20]},
            ]

            for strategy in detection_strategies:
                if strategy['backend'] == 'opencv':
                    faces_data.extend(self.opencv_cascade_detection(enhanced_frame, strategy))
                elif strategy['backend'] == 'mtcnn':
                    faces_data.extend(self.mtcnn_aggressive_detection(enhanced_frame, strategy))

                if faces_data:  # If we found faces, stop trying other strategies
                    break

        except Exception as e:
            logger.error(f"Aggressive detection failed: {e}")

        return faces_data

    def opencv_cascade_detection(self, frame_rgb: np.ndarray, strategy: dict) -> List[Dict]:
        """Enhanced OpenCV cascade detection with multiple parameters."""
        faces_data = []

        try:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            # Load cascade classifiers
            cascade_files = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            ]

            detected_faces = []

            for cascade_file in cascade_files:
                try:
                    face_cascade = cv2.CascadeClassifier(cascade_file)

                    for scale_factor in strategy['scale_factors']:
                        for min_neighbors in strategy['min_neighbors']:
                            faces = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=scale_factor,
                                minNeighbors=min_neighbors,
                                minSize=(15, 15),  # Very small minimum size
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )
                            detected_faces.extend(faces)

                except Exception as e:
                    logger.warning(f"Cascade detection error: {e}")
                    continue

            # Remove duplicates and process faces
            if detected_faces:
                detected_faces = self.remove_duplicate_faces(detected_faces, overlap_threshold=0.2)
                logger.info(f"Aggressive OpenCV found {len(detected_faces)} faces")

                for i, (x, y, w, h) in enumerate(detected_faces):
                    try:
                        # Extract face with quality assessment
                        face_crop = self.extract_high_quality_face(frame_rgb,
                                                                 {'x': x, 'y': y, 'w': w, 'h': h})

                        if face_crop is not None:
                            quality = self.assess_face_quality(face_crop)

                            if quality > 0.35:  # Higher threshold for aggressive detection to reduce false positives
                                # Generate embedding
                                embedding = DeepFace.represent(
                                    img_path=face_crop,
                                    model_name='ArcFace',
                                    detector_backend='skip',
                                    enforce_detection=False,
                                    normalization='ArcFace'
                                )

                                if embedding and len(embedding) > 0:
                                    face_data = {
                                        'face_image': face_crop,
                                        'embedding': embedding[0]['embedding'],
                                        'face_id': str(uuid.uuid4()),
                                        'detection_backend': 'opencv_aggressive',
                                        'embedding_model': 'ArcFace',
                                        'face_area': {'x': x, 'y': y, 'w': w, 'h': h},
                                        'quality_score': quality
                                    }
                                    faces_data.append(face_data)

                                    # Save debug
                                    debug_path = f"debug_face_aggressive_{i}.jpg"
                                    face_uint8 = (face_crop * 255).astype(np.uint8)
                                    cv2.imwrite(debug_path, cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR))

                    except Exception as e:
                        logger.warning(f"Error processing aggressive face {i}: {e}")
                        continue

        except Exception as e:
            logger.error(f"OpenCV aggressive detection failed: {e}")

        return faces_data

    def mtcnn_aggressive_detection(self, frame_rgb: np.ndarray, strategy: dict) -> List[Dict]:
        """MTCNN aggressive detection with multiple face sizes."""
        faces_data = []

        try:
            # Try different MTCNN approaches without unsupported parameters
            logger.info("Attempting MTCNN aggressive detection...")

            try:
                representations = DeepFace.represent(
                    img_path=frame_rgb,
                    model_name='ArcFace',
                    detector_backend='mtcnn',
                    enforce_detection=False,
                    align=True,
                    normalization='ArcFace'
                )

                if representations and len(representations) > 0:
                    logger.info(f"MTCNN aggressive found {len(representations)} faces")

                    for i, representation in enumerate(representations):
                        embedding_vector = representation['embedding']
                        face_area = representation.get('facial_area', {})

                        face_crop = self.extract_high_quality_face(frame_rgb, face_area)

                        if face_crop is not None:
                            quality = self.assess_face_quality(face_crop)

                            if quality > 0.2:  # Lower threshold for aggressive
                                face_data = {
                                    'face_image': face_crop,
                                    'embedding': embedding_vector,
                                    'face_id': str(uuid.uuid4()),
                                    'detection_backend': 'mtcnn_aggressive',
                                    'embedding_model': 'ArcFace',
                                    'face_area': face_area,
                                    'quality_score': quality
                                }
                                faces_data.append(face_data)

                                # Save debug
                                debug_path = f"debug_face_mtcnn_aggressive_{i}.jpg"
                                face_uint8 = (face_crop * 255).astype(np.uint8)
                                cv2.imwrite(debug_path, cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR))

            except Exception as e:
                logger.warning(f"MTCNN aggressive detection failed: {e}")

        except Exception as e:
            logger.error(f"MTCNN aggressive detection failed: {e}")

        return faces_data

    def remove_duplicate_faces(self, faces, overlap_threshold=0.3):
        """Remove overlapping face detections."""
        if len(faces) <= 1:
            return faces

        # Convert to (x, y, x2, y2) format
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])

        boxes = np.array(boxes, dtype=np.float32)

        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Sort by area (keep larger faces)
        indices = np.argsort(areas)[::-1]

        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)

            if len(indices) == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[indices[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[indices[1:], 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            union = areas[i] + areas[indices[1:]] - intersection
            iou = intersection / union

            # Keep faces with low overlap
            indices = indices[1:][iou < overlap_threshold]

        return [faces[i] for i in keep]

    def save_face_image(self, face_image: np.ndarray, face_id: str) -> str:
        """Save cropped face image to disk."""
        face_path = f"data/faces/{face_id}.jpg"
        # Convert from [0,1] range to [0,255] and ensure uint8
        face_image_uint8 = (face_image * 255).astype(np.uint8)
        cv2.imwrite(face_path, cv2.cvtColor(face_image_uint8, cv2.COLOR_RGB2BGR))
        return face_path

    def process_video(self, video_path: str, camera_id: str = "camera_01"):
        """
        Process entire video: extract frames, detect faces, generate embeddings, store in Qdrant.

        Args:
            video_path: Path to the video file
            camera_id: Identifier for the camera/video source
        """
        video_name = os.path.basename(video_path)
        logger.info(f"Starting processing of {video_name}")

        # Create and clear preview directory
        preview_dir = "data/preview_faces"
        if os.path.exists(preview_dir):
            # Clear existing preview images
            for file in os.listdir(preview_dir):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    os.remove(os.path.join(preview_dir, file))
            logger.info("Cleared existing preview images")
        else:
            os.makedirs(preview_dir, exist_ok=True)

        # Extract frames more frequently for better coverage
        frames = self.extract_frames(video_path, frame_interval=30)  # Every 30 frames (1 second at 30fps) for better detection

        points_to_insert = []

        for frame_number, frame in frames:
            logger.info(f"Processing frame {frame_number}")

            # Detect and embed faces
            faces_data = self.detect_and_embed_faces(frame)

            for face_data in faces_data:
                face_id = face_data['face_id']
                embedding = face_data['embedding']
                face_image = face_data['face_image']
                quality_score = face_data.get('quality_score', 0.5)

                # Save face image in data/faces
                face_path = self.save_face_image(face_image, face_id)

                # Save preview image with frame info including model name
                preview_filename = f"frame_{frame_number:04d}_{face_id[:8]}_{face_data.get('detection_backend', 'unknown')}_{face_data.get('embedding_model', 'unknown')}_q{quality_score:.2f}.jpg"
                preview_path = os.path.join(preview_dir, preview_filename)
                face_image_uint8 = (face_image * 255).astype(np.uint8)
                cv2.imwrite(preview_path, cv2.cvtColor(face_image_uint8, cv2.COLOR_RGB2BGR))

                # Prepare metadata with demographics
                metadata = {
                    'face_id': face_id,
                    'video_name': video_name,
                    'frame_number': frame_number,
                    'camera_id': camera_id,
                    'face_path': face_path,
                    'preview_path': preview_path,
                    'timestamp': frame_number / 30.0,
                    'quality_score': quality_score,
                    'detection_backend': face_data.get('detection_backend', 'unknown'),
                    'embedding_model': face_data.get('embedding_model', 'unknown'),
                    # ADD DEMOGRAPHIC DATA TO METADATA
                    'gender': face_data.get('gender', 'unknown'),
                    'age_range': face_data.get('age_range', 'unknown'),
                    'age': face_data.get('age', 0),
                    'dominant_emotion': face_data.get('dominant_emotion', 'unknown'),
                    'dominant_race': face_data.get('dominant_race', 'unknown')
                }

                # Create point for Qdrant
                point = PointStruct(
                    id=face_id,
                    vector=embedding,
                    payload=metadata
                )

                points_to_insert.append(point)
                self.metadata.append(metadata)

                logger.info(f"Prepared high-quality face embedding for frame {frame_number}, face_id: {face_id}, quality: {quality_score:.3f}")

        # Insert all points into Qdrant
        if points_to_insert:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points_to_insert
            )
            logger.info(f"Inserted {len(points_to_insert)} high-quality face embeddings into Qdrant")

            # Save metadata to JSON file
            with open('data/metadata.json', 'w') as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(f"Saved metadata for {len(self.metadata)} high-quality faces")
            logger.info(f"Preview images saved in: {preview_dir}")
        else:
            logger.warning("No high-quality faces detected in the video")

    def get_collection_info(self):
        """Get information about the faces collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

    def analyze_face_demographics(self, face_crop: np.ndarray) -> dict:
        """
        Analyze face demographics using DeepFace's built-in analysis capabilities.

        Args:
            face_crop: Cropped face image as numpy array (0-1 range)

        Returns:
            Dictionary with demographic information
        """
        demographics = {
            'gender': 'unknown',
            'age_range': 'unknown',
            'dominant_emotion': 'unknown',
            'dominant_race': 'unknown'
        }

        try:
            # Convert face crop to suitable format for DeepFace analysis
            face_uint8 = (face_crop * 255).astype(np.uint8)

            # Use DeepFace to analyze demographics
            analysis = DeepFace.analyze(
                img_path=face_uint8,
                actions=['age', 'gender', 'race', 'emotion'],
                detector_backend='skip',  # Skip detection since we already have the face
                enforce_detection=False
            )

            # Extract demographic information from analysis results
            if isinstance(analysis, list) and len(analysis) > 0:
                analysis_result = analysis[0]
            else:
                analysis_result = analysis

            # Extract gender information
            gender_data = analysis_result.get('gender', {})
            if isinstance(gender_data, dict):
                # Get the gender with highest confidence - KEEP ORIGINAL FORMAT
                raw_gender = max(gender_data, key=gender_data.get).lower()
                # CRITICAL FIX: Keep original DeepFace format for consistency
                if raw_gender == 'woman':
                    gender = 'woman'  # Keep as 'woman' not 'female'
                elif raw_gender == 'man':
                    gender = 'man'    # Keep as 'man' not 'male'
                else:
                    gender = raw_gender
                demographics['gender'] = gender

            # Extract age information
            age = analysis_result.get('age', 0)
            if age > 0:
                # Categorize age into ranges for better matching
                if age < 18:
                    age_range = 'child'
                elif age < 30:
                    age_range = 'young_adult'
                elif age < 50:
                    age_range = 'adult'
                elif age < 70:
                    age_range = 'middle_aged'
                else:
                    age_range = 'elderly'
                demographics['age_range'] = age_range
                demographics['age'] = age

            # Extract emotion information
            emotion_data = analysis_result.get('emotion', {})
            if isinstance(emotion_data, dict):
                dominant_emotion = max(emotion_data, key=emotion_data.get).lower()
                demographics['dominant_emotion'] = dominant_emotion

            # Extract race/ethnicity information
            race_data = analysis_result.get('race', {})
            if isinstance(race_data, dict):
                dominant_race = max(race_data, key=race_data.get).lower()
                demographics['dominant_race'] = dominant_race

            logger.info(f"Demographics extracted: Gender={demographics['gender']}, Age={demographics.get('age', 'unknown')}, Emotion={demographics['dominant_emotion']}, Race={demographics['dominant_race']}")

        except Exception as e:
            logger.warning(f"Demographic analysis failed: {e}")
            # Keep default unknown values

        return demographics

def main():
    """Main function to process a sample video."""
    # Initialize ingestor
    ingestor = FaceIngestor()

    # Check if sample video exists
    video_path = "sample.mp4"
    if not os.path.exists(video_path):
        logger.error(f"Sample video not found: {video_path}")
        logger.info("Please add a sample.mp4 file to the project directory")
        return

    # Process the video
    ingestor.process_video(video_path)

    # Show collection info
    info = ingestor.get_collection_info()
    if info:
        logger.info(f"Collection info: {info}")

if __name__ == "__main__":
    main()
