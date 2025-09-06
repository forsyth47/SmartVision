#!/usr/bin/env python3
"""
Professional Face Recognition Engine
Based on best practices from SCRFD+ArcFace and FaceRec systems.

This engine provides proper similarity calculation with normalized embeddings
and appropriate thresholds to prevent false matches.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from deepface import DeepFace
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfessionalFaceRecognitionEngine:
    """
    Professional-grade face recognition engine with proper similarity thresholds.

    Key improvements:
    1. Proper L2 normalization of embeddings
    2. Cosine similarity calculation instead of raw distance
    3. Stricter similarity thresholds (0.4-0.7 range)
    4. Better face quality assessment
    5. Demographic-aware matching
    """

    def __init__(self, db_path: str = "./qdrant_db", collection_name: str = "faces"):
        self.client = QdrantClient(path=db_path)
        self.collection_name = collection_name
        self.vector_size = 512

        # CRITICAL: Professional similarity thresholds
        self.similarity_thresholds = {
            'strict': 0.65,      # Very high confidence matches
            'normal': 0.55,      # Good matches
            'loose': 0.45,       # Acceptable matches
            'minimum': 0.35      # Lowest acceptable threshold
        }

        # Initialize collection if it doesn't exist
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the collection exists with proper configuration."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Properly normalize embedding using L2 norm for cosine similarity.

        Args:
            embedding: Raw embedding vector

        Returns:
            L2-normalized embedding
        """
        embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)

        if norm == 0:
            logger.warning("Zero norm embedding detected, returning zeros")
            return embedding

        return embedding / norm

    def generate_embedding(self, image: np.ndarray, model_name: str = 'ArcFace') -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Generate a properly normalized embedding for face recognition.

        Args:
            image: Input image as numpy array
            model_name: Model to use ('ArcFace', 'Facenet512', etc.)

        Returns:
            Tuple of (normalized_embedding, face_metadata) or None
        """
        try:
            # Ensure proper image format
            if len(image.shape) == 3 and image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[-1] == 3:
                if image.max() > 1:
                    image = image.astype(np.float32) / 255.0
                else:
                    image = image.astype(np.float32)

            # Apply image enhancement
            image_enhanced = self._enhance_image_quality(image)

            # Try multiple detection backends for robustness
            detection_configs = [
                {'backend': 'retinaface', 'model': model_name},
                {'backend': 'mtcnn', 'model': model_name},
                {'backend': 'opencv', 'model': model_name},
            ]

            for config in detection_configs:
                try:
                    logger.info(f"Trying {config['backend']} + {config['model']}...")

                    # Generate representation
                    representations = DeepFace.represent(
                        img_path=image_enhanced,
                        model_name=config['model'],
                        detector_backend=config['backend'],
                        enforce_detection=False,
                        align=True,
                        normalization='ArcFace'
                    )

                    if representations and len(representations) > 0:
                        embedding = np.array(representations[0]['embedding'], dtype=np.float32)

                        # CRITICAL: Proper L2 normalization
                        normalized_embedding = self.normalize_embedding(embedding)

                        # Generate demographic data for matching
                        demographics = self._analyze_demographics(image_enhanced)

                        # Assess face quality
                        quality_score = self._assess_face_quality(image_enhanced)

                        metadata = {
                            'demographics': demographics,
                            'quality_score': quality_score,
                            'detection_backend': config['backend'],
                            'model': config['model'],
                            'embedding_norm': float(np.linalg.norm(embedding))
                        }

                        logger.info(f"✅ Generated embedding: norm={metadata['embedding_norm']:.3f}, "
                                  f"quality={quality_score:.3f}, gender={demographics.get('gender', 'unknown')}")

                        return normalized_embedding, metadata

                except Exception as e:
                    logger.warning(f"❌ {config['backend']} + {config['model']} failed: {e}")
                    continue

            logger.error("❌ All embedding generation methods failed")
            return None

        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            return None

    def search_faces(self, query_embedding: np.ndarray, query_metadata: Dict = None,
                    threshold_level: str = 'normal', max_results: int = 20) -> List[Dict]:
        """
        Search for similar faces using proper cosine similarity thresholds.

        Args:
            query_embedding: Normalized query embedding
            query_metadata: Query face metadata
            threshold_level: Threshold level ('strict', 'normal', 'loose', 'minimum')
            max_results: Maximum results to return

        Returns:
            List of search results with proper similarity scores
        """
        try:
            # Get similarity threshold
            similarity_threshold = self.similarity_thresholds.get(threshold_level, 0.55)

            # Convert similarity threshold to distance threshold for Qdrant
            # For cosine distance: distance = 1 - similarity
            distance_threshold = 1.0 - similarity_threshold

            # Search in vector database
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=max_results * 2,  # Get more candidates for filtering
                with_payload=True
            )

            processed_results = []

            for result in search_results:
                # Extract result data
                if hasattr(result, 'score'):
                    distance = result.score
                    payload = result.payload
                    point_id = result.id
                else:
                    distance = result.get('score', 1.0)
                    payload = result.get('payload', {})
                    point_id = result.get('id', '')

                # CRITICAL: Proper cosine similarity calculation
                cosine_similarity = 1.0 - distance

                # Apply strict threshold filtering
                if cosine_similarity < similarity_threshold:
                    continue

                # Enhanced demographic filtering
                adjusted_similarity = self._apply_demographic_filtering(
                    cosine_similarity, query_metadata, payload
                )

                # Quality-based filtering
                quality_score = payload.get('quality_score', 0.5)
                if quality_score < 0.3:  # Skip very low quality faces
                    adjusted_similarity *= 0.8  # Penalize low quality

                result_dict = {
                    'similarity': float(cosine_similarity),
                    'adjusted_similarity': float(adjusted_similarity),
                    'distance': float(distance),
                    'payload': payload,
                    'point_id': str(point_id),
                    'threshold_level': threshold_level,
                    'passes_threshold': adjusted_similarity >= similarity_threshold
                }

                processed_results.append(result_dict)

            # Sort by adjusted similarity
            processed_results.sort(key=lambda x: x['adjusted_similarity'], reverse=True)

            # Final filtering - only return results that pass threshold
            filtered_results = [r for r in processed_results if r['passes_threshold']]

            logger.info(f"Search completed: {len(filtered_results)} results above {similarity_threshold:.2f} threshold")

            # Log top results
            if filtered_results:
                logger.info(f"Top result: similarity={filtered_results[0]['similarity']:.3f}, "
                          f"adjusted={filtered_results[0]['adjusted_similarity']:.3f}")

            return filtered_results[:max_results]

        except Exception as e:
            logger.error(f"Error in face search: {e}")
            return []

    def _apply_demographic_filtering(self, base_similarity: float, query_metadata: Dict,
                                   result_payload: Dict) -> float:
        """
        Apply demographic-aware filtering to similarity scores.

        Args:
            base_similarity: Base cosine similarity score
            query_metadata: Query face metadata
            result_payload: Result face payload

        Returns:
            Adjusted similarity score
        """
        if not query_metadata or 'demographics' not in query_metadata:
            return base_similarity

        query_demographics = query_metadata['demographics']
        adjusted_similarity = base_similarity

        # Strong gender matching
        query_gender = query_demographics.get('gender', 'unknown')
        result_gender = result_payload.get('gender', 'unknown')

        if query_gender != 'unknown' and result_gender != 'unknown':
            if query_gender == result_gender:
                adjusted_similarity += 0.02  # Small boost for same gender
            else:
                adjusted_similarity -= 0.12  # Strong penalty for different gender

        # Age range matching
        query_age_range = query_demographics.get('age_range', 'unknown')
        result_age_range = result_payload.get('age_range', 'unknown')

        if query_age_range != 'unknown' and result_age_range != 'unknown':
            if query_age_range == result_age_range:
                adjusted_similarity += 0.01
            else:
                adjusted_similarity -= 0.05

        # Ensure adjusted similarity is within valid range
        return max(0.0, min(1.0, adjusted_similarity))

    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better face recognition.

        Args:
            image: Input image (0-1 range)

        Returns:
            Enhanced image
        """
        try:
            # Convert to uint8 for processing
            image_uint8 = (image * 255).astype(np.uint8)

            # Apply CLAHE for better contrast
            lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)

            enhanced_lab = cv2.merge([l, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

            # Apply gentle noise reduction
            enhanced_rgb = cv2.bilateralFilter(enhanced_rgb, 5, 50, 50)

            return enhanced_rgb.astype(np.float32) / 255.0

        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image

    def _assess_face_quality(self, image: np.ndarray) -> float:
        """
        Assess face quality using multiple metrics.

        Args:
            image: Face image (0-1 range)

        Returns:
            Quality score (0-1, higher is better)
        """
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                face_uint8 = (image * 255).astype(np.uint8)
                gray = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2GRAY)
            else:
                gray = (image * 255).astype(np.uint8)

            # Calculate sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)

            # Calculate contrast
            contrast = gray.std() / 255.0

            # Calculate brightness balance
            brightness = gray.mean() / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2

            # Combined quality score
            quality = (sharpness_score * 0.5 + contrast * 0.3 + brightness_score * 0.2)

            return max(0.0, min(1.0, quality))

        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5

    def _analyze_demographics(self, image: np.ndarray) -> Dict:
        """
        Analyze face demographics.

        Args:
            image: Face image

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
            face_uint8 = (image * 255).astype(np.uint8)

            analysis = DeepFace.analyze(
                img_path=face_uint8,
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
            logger.warning(f"Demographic analysis failed: {e}")

        return demographics

    def get_threshold_info(self) -> Dict:
        """Get information about similarity thresholds."""
        return {
            'thresholds': self.similarity_thresholds,
            'recommendations': {
                'strict': 'Use for high-security applications (few false positives)',
                'normal': 'Balanced approach for most applications',
                'loose': 'More permissive, higher recall but more false positives',
                'minimum': 'Very permissive, use with caution'
            }
        }
