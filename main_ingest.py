#!/usr/bin/env python3
"""
Professional Face Ingestion Engine
Based on SCRFD+ArcFace and FaceRec architectures with proper embedding normalization.

This system creates a properly normalized face database suitable for accurate recognition.
"""

import cv2
import os
import json
import uuid
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from deepface import DeepFace
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProfessionalFaceIngestor:
    """
    Professional face ingestion system with proper embedding normalization.

    Key Features:
    1. L2-normalized embeddings for accurate cosine similarity
    2. Strict quality thresholds to prevent low-quality faces
    3. Professional similarity-ready database
    4. Enhanced face detection and preprocessing
    """

    def __init__(self, db_path: str = "./qdrant_db", collection_name: str = "faces"):
        self.client = QdrantClient(path=db_path)
        self.collection_name = collection_name
        self.vector_size = 512

        # Professional quality thresholds
        self.min_quality_threshold = 0.4  # Minimum face quality
        self.min_face_size = 40  # Minimum face size in pixels
        self.max_faces_per_frame = 5  # Limit faces per frame to best quality ones

        # Initialize fresh collection
        self._setup_collection()

        # Create directories
        os.makedirs("data/faces", exist_ok=True)
        os.makedirs("data/preview_faces", exist_ok=True)

        self.metadata = []
        self.processed_faces = 0

    def _setup_collection(self):
        """Setup fresh collection with proper configuration."""
        try:
            # Delete existing collection for fresh start
            try:
                self.client.delete_collection(self.collection_name)
                logger.info("🗑️ Cleared existing collection for fresh professional ingestion")
            except Exception:
                logger.info("No existing collection to clear")

            # Create fresh collection with cosine distance
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            logger.info("✅ Created fresh professional collection with cosine distance")

        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding using L2 norm for proper cosine similarity.

        Args:
            embedding: Raw embedding vector

        Returns:
            L2-normalized embedding
        """
        embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)

        if norm == 0:
            logger.warning("Zero norm embedding detected")
            return embedding

        normalized = embedding / norm

        # Verify normalization
        new_norm = np.linalg.norm(normalized)
        if abs(new_norm - 1.0) > 1e-6:
            logger.warning(f"Normalization verification failed: norm = {new_norm}")

        return normalized

    def extract_frames(self, video_path: str, frame_interval: int = 30) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames from video with smart sampling.

        Args:
            video_path: Path to video file
            frame_interval: Extract every N frames (30 = 1 second at 30fps)

        Returns:
            List of (frame_number, frame_image) tuples
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}, Interval: {frame_interval}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps if fps > 0 else 0
                frames.append((frame_count, frame, timestamp))

            frame_count += 1

            # Progress indicator
            if frame_count % 1000 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames

    def detect_and_process_faces(self, frame: np.ndarray, frame_number: int,
                               timestamp: float) -> List[Dict]:
        """
        Detect faces and generate professional embeddings.

        Args:
            frame: OpenCV frame (BGR)
            frame_number: Frame number in video
            timestamp: Timestamp in seconds

        Returns:
            List of processed face data
        """
        faces_data = []

        try:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Enhanced preprocessing
            frame_enhanced = self._enhance_frame(frame_rgb)

            # Professional detection pipeline
            detection_configs = [
                {'backend': 'retinaface', 'model': 'ArcFace'},
                {'backend': 'mtcnn', 'model': 'ArcFace'},
                {'backend': 'opencv', 'model': 'ArcFace'},
            ]

            best_faces = []

            for config in detection_configs:
                try:
                    logger.debug(f"Trying {config['backend']} + {config['model']}...")

                    representations = DeepFace.represent(
                        img_path=frame_enhanced,
                        model_name=config['model'],
                        detector_backend=config['backend'],
                        enforce_detection=False,
                        align=True,
                        normalization='ArcFace'
                    )

                    if representations and len(representations) > 0:
                        for i, repr_data in enumerate(representations):
                            embedding = np.array(repr_data['embedding'], dtype=np.float32)
                            face_area = repr_data.get('facial_area', {})

                            # Extract and assess face
                            face_crop = self._extract_face_crop(frame_enhanced, face_area)
                            if face_crop is None:
                                continue

                            # Professional quality assessment
                            quality_metrics = self._assess_face_quality_professional(face_crop)

                            if quality_metrics['overall_score'] >= self.min_quality_threshold:
                                # CRITICAL: Professional embedding normalization
                                normalized_embedding = self.normalize_embedding(embedding)

                                # Demographic analysis
                                demographics = self._analyze_demographics(face_crop)

                                face_data = {
                                    'embedding': normalized_embedding,
                                    'face_crop': face_crop,
                                    'face_area': face_area,
                                    'quality_metrics': quality_metrics,
                                    'demographics': demographics,
                                    'detection_config': config,
                                    'frame_info': {
                                        'frame_number': frame_number,
                                        'timestamp': timestamp
                                    }
                                }

                                best_faces.append(face_data)

                                logger.debug(f"✅ High-quality face found: "
                                           f"quality={quality_metrics['overall_score']:.3f}, "
                                           f"gender={demographics.get('gender', 'unknown')}")

                        # If we found good faces with this config, use them
                        if best_faces:
                            break

                except Exception as e:
                    logger.warning(f"Detection failed with {config['backend']}: {e}")
                    continue

            # Select best faces (limit per frame)
            if best_faces:
                # Sort by quality and take best ones
                best_faces.sort(key=lambda x: x['quality_metrics']['overall_score'], reverse=True)
                selected_faces = best_faces[:self.max_faces_per_frame]

                # Process selected faces
                for face_data in selected_faces:
                    processed_face = self._finalize_face_data(face_data, frame_number, timestamp)
                    if processed_face:
                        faces_data.append(processed_face)

            if faces_data:
                logger.info(f"Frame {frame_number}: Found {len(faces_data)} high-quality faces")

        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return faces_data

    def _enhance_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Professional frame enhancement for better face detection.

        Args:
            frame_rgb: RGB frame

        Returns:
            Enhanced frame
        """
        try:
            # Convert to LAB for better enhancement
            lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge and convert back
            enhanced_lab = cv2.merge([l, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

            # Gentle noise reduction
            enhanced_rgb = cv2.bilateralFilter(enhanced_rgb, 5, 50, 50)

            return enhanced_rgb

        except Exception as e:
            logger.warning(f"Frame enhancement failed: {e}")
            return frame_rgb

    def _extract_face_crop(self, frame: np.ndarray, face_area: Dict,
                          padding: float = 0.3) -> Optional[np.ndarray]:
        """
        Extract face crop with professional standards.

        Args:
            frame: RGB frame
            face_area: Face area coordinates
            padding: Padding around face

        Returns:
            Face crop or None
        """
        try:
            if not face_area or not all(k in face_area for k in ['x', 'y', 'w', 'h']):
                return None

            x, y, w, h = [int(face_area[k]) for k in ['x', 'y', 'w', 'h']]

            # Check minimum face size
            if min(w, h) < self.min_face_size:
                logger.debug(f"Face too small: {w}x{h} < {self.min_face_size}")
                return None

            # Add padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)

            if x2 <= x1 or y2 <= y1:
                return None

            face_crop = frame[y1:y2, x1:x2]

            # Resize to standard size
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_CUBIC)
                return face_crop

        except Exception as e:
            logger.warning(f"Face crop extraction failed: {e}")

        return None

    def _assess_face_quality_professional(self, face_crop: np.ndarray) -> Dict:
        """
        Professional face quality assessment with multiple metrics.

        Args:
            face_crop: Face crop image

        Returns:
            Quality metrics dictionary
        """
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)

            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)

            # Contrast
            contrast = gray.std() / 255.0

            # Brightness balance
            brightness = gray.mean() / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2

            # Face size score
            size_score = min(min(face_crop.shape[:2]) / 112.0, 1.0)

            # Edge density (face structure)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(edge_density * 10, 1.0)

            # Combined score with professional weights
            overall_score = (
                sharpness_score * 0.35 +
                contrast * 0.25 +
                brightness_score * 0.2 +
                size_score * 0.1 +
                edge_score * 0.1
            )

            return {
                'overall_score': max(0.0, min(1.0, overall_score)),
                'sharpness': sharpness_score,
                'contrast': contrast,
                'brightness': brightness_score,
                'size_score': size_score,
                'edge_score': edge_score
            }

        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {'overall_score': 0.3}

    def _analyze_demographics(self, face_crop: np.ndarray) -> Dict:
        """
        Analyze face demographics consistently.

        Args:
            face_crop: Face crop image

        Returns:
            Demographics dictionary
        """
        demographics = {
            'gender': 'unknown',
            'age_range': 'unknown',
            'dominant_emotion': 'unknown',
            'dominant_race': 'unknown'
        }

        try:
            analysis = DeepFace.analyze(
                img_path=face_crop,
                actions=['age', 'gender', 'race', 'emotion'],
                detector_backend='skip',
                enforce_detection=False
            )

            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]

            # Extract demographics
            gender_data = analysis.get('gender', {})
            if isinstance(gender_data, dict):
                raw_gender = max(gender_data, key=gender_data.get).lower()
                demographics['gender'] = 'woman' if raw_gender == 'woman' else 'man' if raw_gender == 'man' else raw_gender

            age = analysis.get('age', 0)
            if age > 0:
                if age < 18:
                    demographics['age_range'] = 'child'
                elif age < 30:
                    demographics['age_range'] = 'young_adult'
                elif age < 50:
                    demographics['age_range'] = 'adult'
                elif age < 70:
                    demographics['age_range'] = 'middle_aged'
                else:
                    demographics['age_range'] = 'elderly'
                demographics['age'] = age

            emotion_data = analysis.get('emotion', {})
            if isinstance(emotion_data, dict):
                demographics['dominant_emotion'] = max(emotion_data, key=emotion_data.get).lower()

            race_data = analysis.get('race', {})
            if isinstance(race_data, dict):
                demographics['dominant_race'] = max(race_data, key=race_data.get).lower()

        except Exception as e:
            logger.debug(f"Demographic analysis failed: {e}")

        return demographics

    def _finalize_face_data(self, face_data: Dict, frame_number: int, timestamp: float) -> Optional[Dict]:
        """
        Finalize face data for storage.

        Args:
            face_data: Processed face data
            frame_number: Frame number
            timestamp: Timestamp

        Returns:
            Finalized face data ready for storage
        """
        try:
            face_id = str(uuid.uuid4())

            # Save face image
            face_path = f"data/faces/{face_id}.jpg"
            face_crop_uint8 = (face_data['face_crop'] * 255).astype(np.uint8)
            cv2.imwrite(face_path, cv2.cvtColor(face_crop_uint8, cv2.COLOR_RGB2BGR))

            # Save preview image with metadata
            quality_score = face_data['quality_metrics']['overall_score']
            detection_backend = face_data['detection_config']['backend']
            model = face_data['detection_config']['model']

            preview_filename = f"frame_{frame_number:04d}_{face_id[:8]}_{detection_backend}_{model}_q{quality_score:.2f}.jpg"
            preview_path = f"data/preview_faces/{preview_filename}"
            cv2.imwrite(preview_path, cv2.cvtColor(face_crop_uint8, cv2.COLOR_RGB2BGR))

            # Prepare metadata
            metadata = {
                'face_id': face_id,
                'face_path': face_path,
                'preview_path': preview_path,
                'frame_number': frame_number,
                'timestamp': timestamp,
                'video_name': 'sample.mp4',  # Will be updated by caller
                'camera_id': 'camera_01',   # Will be updated by caller

                # Quality metrics
                'quality_score': quality_score,
                'quality_metrics': face_data['quality_metrics'],

                # Detection info
                'detection_backend': detection_backend,
                'embedding_model': model,

                # Demographics
                **face_data['demographics'],

                # Professional embedding info
                'embedding_norm': float(np.linalg.norm(face_data['embedding'])),
                'is_normalized': True
            }

            # Verify embedding normalization
            if abs(metadata['embedding_norm'] - 1.0) > 1e-6:
                logger.warning(f"Embedding not properly normalized: norm={metadata['embedding_norm']}")

            return {
                'metadata': metadata,
                'embedding': face_data['embedding']
            }

        except Exception as e:
            logger.error(f"Failed to finalize face data: {e}")
            return None

    def process_video_professional(self, video_path: str, camera_id: str = "camera_01"):
        """
        Process video with professional standards.

        Args:
            video_path: Path to video file
            camera_id: Camera identifier
        """
        video_name = os.path.basename(video_path)
        logger.info(f"🎬 Starting professional processing of {video_name}")

        # Clear preview directory
        preview_dir = "data/preview_faces"
        if os.path.exists(preview_dir):
            for file in os.listdir(preview_dir):
                if file.endswith('.jpg'):
                    os.remove(os.path.join(preview_dir, file))

        # Extract frames
        frames = self.extract_frames(video_path, frame_interval=30)

        points_to_insert = []
        total_faces = 0

        for frame_number, frame, timestamp in frames:
            logger.info(f"Processing frame {frame_number} (timestamp: {timestamp:.1f}s)")

            faces_data = self.detect_and_process_faces(frame, frame_number, timestamp)

            for face_data in faces_data:
                # Update metadata with video info
                face_data['metadata']['video_name'] = video_name
                face_data['metadata']['camera_id'] = camera_id

                # Store metadata
                self.metadata.append(face_data['metadata'])

                # Prepare point for Qdrant
                point = PointStruct(
                    id=face_data['metadata']['face_id'],
                    vector=face_data['embedding'].tolist(),
                    payload=face_data['metadata']
                )
                points_to_insert.append(point)

                total_faces += 1

                # Log progress
                if total_faces % 10 == 0:
                    logger.info(f"Processed {total_faces} faces so far...")

        # Insert into database
        if points_to_insert:
            logger.info(f"💾 Inserting {len(points_to_insert)} professional embeddings into database...")

            # Insert in batches for better performance
            batch_size = 100
            for i in range(0, len(points_to_insert), batch_size):
                batch = points_to_insert[i:i + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(points_to_insert)-1)//batch_size + 1}")

            # Save metadata
            with open('data/metadata.json', 'w') as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(f"✅ Professional ingestion complete!")
            logger.info(f"📊 Statistics:")
            logger.info(f"  - Total frames processed: {len(frames)}")
            logger.info(f"  - High-quality faces found: {total_faces}")
            logger.info(f"  - Average faces per frame: {total_faces/len(frames):.2f}")
            logger.info(f"  - All embeddings are L2-normalized for professional similarity calculation")

            # Quality distribution
            quality_scores = [face['quality_score'] for face in self.metadata]
            avg_quality = np.mean(quality_scores)
            min_quality = np.min(quality_scores)
            max_quality = np.max(quality_scores)

            logger.info(f"📈 Quality Distribution:")
            logger.info(f"  - Average quality: {avg_quality:.3f}")
            logger.info(f"  - Quality range: {min_quality:.3f} - {max_quality:.3f}")
            logger.info(f"  - Faces above 0.6 quality: {sum(1 for q in quality_scores if q > 0.6)}")

        else:
            logger.warning("❌ No high-quality faces found in video!")

    def get_collection_info(self):
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None


def main():
    """Main function for professional face ingestion."""
    # Initialize professional ingestor
    ingestor = ProfessionalFaceIngestor()

    # Check video
    video_path = "sample.mp4"
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        logger.info("Please add a sample.mp4 file to the project directory")
        return

    logger.info("🚀 Starting PROFESSIONAL face ingestion with proper embedding normalization")
    logger.info("This will create a database suitable for accurate similarity matching")

    # Process video with professional standards
    ingestor.process_video_professional(video_path)

    # Show collection info
    info = ingestor.get_collection_info()
    if info:
        logger.info(f"Final collection info: {info}")

    logger.info("🎉 Professional database is ready for accurate face recognition!")


if __name__ == "__main__":
    main()
