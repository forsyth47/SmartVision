import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
from qdrant_client import QdrantClient
import json
import os
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceSearchApp:
    def __init__(self):
        """Initialize the face search application."""
        # Use the same persistent database as ingestion
        self.client = QdrantClient(path="./qdrant_db")  # Same path as ingest.py
        self.collection_name = "faces"
        self.load_metadata()

    def load_metadata(self):
        """Load metadata from JSON file."""
        try:
            with open('data/metadata.json', 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.metadata)} faces")
        except FileNotFoundError:
            self.metadata = []
            logger.warning("No metadata file found. Please run ingest.py first.")

    def generate_query_embedding(self, image: np.ndarray) -> list:
        """
        Generate embedding for the query image using EXACTLY the same enhanced method as ingestion.

        Args:
            image: Input image as numpy array

        Returns:
            512D embedding vector
        """
        try:
            # Convert RGBA to RGB if necessary
            if len(image.shape) == 3 and image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                logger.info("Converted RGBA image to RGB")
            elif len(image.shape) == 3 and image.shape[-1] == 3:
                # Ensure RGB format
                if image.max() > 1:
                    image = image.astype(np.float32) / 255.0
                else:
                    image = image.astype(np.float32)

            # CRITICAL: Apply the SAME enhancements as ingestion
            image_enhanced = self.enhance_query_image(image)

            logger.info(f"Query image enhanced and resized to: {image_enhanced.shape}")

            # Use EXACTLY the same configuration as ingestion (best models first)
            detection_configs = [
                {'backend': 'retinaface', 'model': 'ArcFace'},
                {'backend': 'mtcnn', 'model': 'ArcFace'},
                {'backend': 'retinaface', 'model': 'Facenet512'},
                {'backend': 'mtcnn', 'model': 'Facenet512'},
                {'backend': 'opencv', 'model': 'ArcFace'},
                {'backend': 'ssd', 'model': 'ArcFace'},
            ]

            for config in detection_configs:
                try:
                    logger.info(f"Generating embedding with {config['backend']} + {config['model']}...")

                    representations = DeepFace.represent(
                        img_path=image_enhanced,
                        model_name=config['model'],
                        detector_backend=config['backend'],
                        enforce_detection=False,
                        align=True,
                        normalization='ArcFace'
                    )

                    if representations and len(representations) > 0:
                        # CRITICAL: Extract face the SAME way as ingestion
                        for i, representation in enumerate(representations):
                            embedding_vector = representation['embedding']
                            face_area = representation.get('facial_area', {})

                            # Use the SAME face extraction method as ingestion
                            face_crop = self.extract_high_quality_face(image_enhanced, face_area)

                            if face_crop is not None:
                                # Assess quality the same way
                                quality_score = self.assess_face_quality(face_crop)

                                # CRITICAL: Apply the same quality normalization as ingestion
                                normalized_quality = self.normalize_quality_for_bias_prevention(quality_score)

                                logger.info(f"Query face quality score: {quality_score:.3f}, normalized: {normalized_quality:.3f}")

                                if normalized_quality > 0.25:  # Same threshold as ingestion
                                    # CRITICAL: Re-generate embedding with the extracted high-quality face
                                    # This ensures consistency with the ingestion process
                                    final_embedding = DeepFace.represent(
                                        img_path=face_crop,
                                        model_name=config['model'],
                                        detector_backend='skip',  # Skip detection since we have the face
                                        enforce_detection=False,
                                        normalization='ArcFace'
                                    )

                                    if final_embedding and len(final_embedding) > 0:
                                        embedding_vector = final_embedding[0]['embedding']

                                        # DEMOGRAPHIC ANALYSIS: Extract demographic information for query image
                                        query_demographics = self.analyze_query_demographics(face_crop)
                                        logger.info(f"Query demographics: {query_demographics}")

                                        logger.info(f"✅ Successfully generated query embedding with {config['backend']} + {config['model']}")
                                        logger.info(f"Embedding vector length: {len(embedding_vector)}")
                                        logger.info(f"Embedding stats: min={np.min(embedding_vector):.4f}, max={np.max(embedding_vector):.4f}, mean={np.mean(embedding_vector):.4f}")
                                        return embedding_vector, query_demographics

                except Exception as e:
                    logger.warning(f"❌ {config['backend']} + {config['model']} failed: {str(e)}")
                    continue

            # CRITICAL: Add aggressive detection fallback just like ingestion
            logger.info("No high-quality faces found with standard detection, trying aggressive detection...")
            return self.aggressive_face_detection(image_enhanced)

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def enhance_query_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the same enhancements to query image as used in ingestion.

        Args:
            image: Input image (0-1 range)

        Returns:
            Enhanced image
        """
        try:
            # Convert to uint8 for processing
            image_uint8 = (image * 255).astype(np.uint8)

            # Apply the same enhancement pipeline as ingestion
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE (same as ingestion)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)

            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge([l, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

            # Apply gentle denoising (same as ingestion)
            enhanced_rgb = cv2.bilateralFilter(enhanced_rgb, 5, 50, 50)

            # Apply face-specific enhancements
            enhanced_rgb = self.enhance_face_region(enhanced_rgb)

            return enhanced_rgb.astype(np.float32) / 255.0

        except Exception as e:
            logger.warning(f"Query image enhancement failed: {e}")
            return image

    def extract_high_quality_face(self, frame_rgb: np.ndarray, face_area: dict, padding: float = 0.4) -> np.ndarray:
        """
        Extract face region with enhanced quality processing - EXACTLY same as ingestion.

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
                        face_crop = self.enhance_face_region_detailed(face_crop)

                        return face_crop.astype(np.float32) / 255.0

            # Fallback: use center region with enhancement
            h, w = frame_rgb.shape[:2]
            center_crop = frame_rgb[h//4:3*h//4, w//4:3*w//4]
            center_crop = cv2.resize(center_crop, (256, 256), interpolation=cv2.INTER_CUBIC)
            center_crop = self.enhance_face_region_detailed(center_crop)
            return center_crop.astype(np.float32) / 255.0

        except Exception as e:
            logger.warning(f"Error extracting high-quality face: {e}")
            return None

    def enhance_face_region(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Apply specific enhancements to face region as used in ingestion.

        Args:
            face_crop: Face region as numpy array

        Returns:
            Enhanced face region
        """
        try:
            # Convert to YUV for better processing
            yuv = cv2.cvtColor(face_crop, cv2.COLOR_RGB2YUV)

            # Enhance Y channel (luminance)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])

            # Convert back to RGB
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

            # Apply unsharp masking for better details
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255)

            return enhanced

        except Exception as e:
            logger.warning(f"Face enhancement failed: {e}")
            return face_crop

    def enhance_face_region_detailed(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Apply specific enhancements to face region - EXACTLY same as ingestion.

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
        Assess the quality of a detected face for filtering - EXACTLY same as ingestion.

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

    def search_similar_faces(self, query_embedding: list, query_demographics: dict = None, similarity_threshold: float = 0.4, max_results: int = 50):
        """
        Search for similar faces in Qdrant with demographic-aware bias prevention.

        Args:
            query_embedding: 512D embedding vector
            query_demographics: Query face demographics for bias prevention
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            max_results: Maximum number of results to return

        Returns:
            Search results from Qdrant filtered by similarity threshold and sorted by similarity
        """
        try:
            # Get more results than needed for better filtering
            search_limit = min(max_results * 3, 200)  # Get more candidates

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                with_payload=True
            )

            # Process results and add similarity scores with enhanced bias prevention
            processed_results = []
            for result in results:
                # Handle both ScoredPoint objects and dict objects
                if hasattr(result, 'score'):
                    score = result.score
                    payload = result.payload
                    point_id = result.id
                else:
                    # Handle dict format
                    score = result.get('score', 1.0)
                    payload = result.get('payload', {})
                    point_id = result.get('id', '')

                similarity_score = 1 - score  # Convert distance to similarity

                # DEBUG: Log similarity conversion
                logger.debug(f"Distance score: {score:.4f} -> Similarity: {similarity_score:.4f}")

                # CRITICAL: Handle edge cases in similarity calculation
                if similarity_score < 0:
                    similarity_score = 0.0
                elif similarity_score > 1:
                    similarity_score = 1.0

                # ENHANCED BIAS PREVENTION AND SIMILARITY CALCULATION
                quality_score = payload.get('quality_score', 0.5)

                # Start with base similarity score
                adjusted_similarity = similarity_score

                # 1. CRITICAL: Strong gender matching requirement
                if query_demographics:
                    query_gender = query_demographics.get('gender', 'unknown')
                    result_gender = payload.get('gender', 'unknown')

                    # VERY STRONG gender matching requirement
                    if query_gender != 'unknown' and result_gender != 'unknown':
                        if query_gender == result_gender:
                            # Small boost for same gender
                            gender_boost = 0.03  # 3% boost for same gender
                            adjusted_similarity += gender_boost
                            logger.debug(f"Gender match boost: query={query_gender}, result={result_gender}")
                        else:
                            # STRONG penalty for different gender to prevent cross-gender false matches
                            gender_penalty = 0.15  # 15% penalty for different gender
                            adjusted_similarity -= gender_penalty
                            logger.debug(f"Gender mismatch penalty: query={query_gender}, result={result_gender}")

                    # Additional demographic matching
                    query_age_range = query_demographics.get('age_range', 'unknown')
                    result_age_range = payload.get('age_range', 'unknown')

                    if query_age_range != 'unknown' and result_age_range != 'unknown':
                        if query_age_range == result_age_range:
                            age_boost = 0.02  # 2% boost for same age range
                            adjusted_similarity += age_boost
                        else:
                            age_penalty = 0.05  # 5% penalty for different age range
                            adjusted_similarity -= age_penalty

                # 2. Quality bias correction (reduce advantage of overly high-quality faces)
                if quality_score > 0.85:
                    bias_penalty = (quality_score - 0.85) * 0.2  # Penalty for very high quality
                    adjusted_similarity -= bias_penalty
                elif quality_score < 0.4:
                    # Small penalty for very low quality to avoid false matches
                    low_quality_penalty = (0.4 - quality_score) * 0.1
                    adjusted_similarity -= low_quality_penalty

                # 3. Diversity promotion: penalize repeated results from same frame
                frame_number = payload.get('frame_number', 0)
                same_frame_count = len([r for r in processed_results if r['payload'].get('frame_number') == frame_number])
                if same_frame_count > 0:
                    repetition_penalty = same_frame_count * 0.02  # Progressive penalty
                    adjusted_similarity -= repetition_penalty

                # Ensure adjusted similarity doesn't exceed 1.0 or go below 0
                adjusted_similarity = max(0.0, min(adjusted_similarity, 1.0))

                # Only include results above threshold
                if adjusted_similarity >= similarity_threshold:
                    result_dict = {
                        'similarity_score': similarity_score,
                        'adjusted_similarity': adjusted_similarity,
                        'distance_score': score,
                        'payload': payload,
                        'point_id': point_id,
                        'original_result': result
                    }
                    processed_results.append(result_dict)

            # Sort by adjusted similarity score (highest first)
            processed_results.sort(key=lambda x: x['adjusted_similarity'], reverse=True)

            # Limit to max_results
            processed_results = processed_results[:max_results]

            logger.info(f"Found {len(processed_results)} faces above {similarity_threshold} threshold (sorted by similarity)")

            # Log top results for debugging
            if processed_results:
                logger.info("Top 3 results with demographic info:")
                for i, result in enumerate(processed_results[:3]):
                    payload = result['payload']
                    logger.info(f"  {i+1}. Similarity: {result['similarity_score']:.4f}, "
                              f"Adjusted: {result['adjusted_similarity']:.4f}, "
                              f"Quality: {payload.get('quality_score', 'N/A')}, "
                              f"Gender: {payload.get('gender', 'unknown')}, "
                              f"Detection: {payload.get('detection_backend', 'N/A')}")

            return processed_results

        except Exception as e:
            logger.error(f"Error searching faces: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def load_face_image(self, face_path: str):
        """Load and return face image from path."""
        if os.path.exists(face_path):
            return Image.open(face_path)
        return None

    def aggressive_face_detection(self, frame_rgb: np.ndarray) -> list:
        """
        Aggressive face detection for difficult cases using multiple approaches - EXACTLY same as ingestion.
        """
        try:
            # Try OpenCV cascade detection first
            gray = cv2.cvtColor((frame_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Load cascade classifiers
            cascade_files = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            ]

            detected_faces = []
            for cascade_file in cascade_files:
                try:
                    face_cascade = cv2.CascadeClassifier(cascade_file)

                    # Try multiple parameters
                    for scale_factor in [1.05, 1.1, 1.2]:
                        for min_neighbors in [1, 2, 3]:
                            faces = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=scale_factor,
                                minNeighbors=min_neighbors,
                                minSize=(15, 15),
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )
                            detected_faces.extend(faces)
                except Exception as e:
                    logger.warning(f"Cascade detection error: {e}")
                    continue

            # Remove duplicates
            if detected_faces:
                detected_faces = self.remove_duplicate_faces(detected_faces)
                logger.info(f"Aggressive detection found {len(detected_faces)} faces")

                for i, (x, y, w, h) in enumerate(detected_faces):
                    try:
                        # Extract face with quality assessment
                        face_crop = self.extract_high_quality_face(frame_rgb,
                                                                 {'x': x, 'y': y, 'w': w, 'h': h})

                        if face_crop is not None:
                            quality = self.assess_face_quality(face_crop)
                            normalized_quality = self.normalize_quality_for_bias_prevention(quality)

                            logger.info(f"Aggressive detection face {i} - quality: {quality:.3f}, normalized: {normalized_quality:.3f}")

                            if normalized_quality > 0.2:  # Lower threshold for aggressive detection
                                # Generate embedding
                                embedding = DeepFace.represent(
                                    img_path=face_crop,
                                    model_name='ArcFace',
                                    detector_backend='skip',
                                    enforce_detection=False,
                                    normalization='ArcFace'
                                )

                                if embedding and len(embedding) > 0:
                                    embedding_vector = embedding[0]['embedding']

                                    # CRITICAL: Analyze demographics for aggressive detection too
                                    query_demographics = self.analyze_query_demographics(face_crop)

                                    logger.info(f"✅ Aggressive detection generated embedding with quality {quality:.3f}, normalized {normalized_quality:.3f}")
                                    return embedding_vector, query_demographics

                    except Exception as e:
                        logger.warning(f"Error processing aggressive face {i}: {e}")
                        continue

            logger.error("❌ All detection methods failed for query image")
            return None

        except Exception as e:
            logger.error(f"Aggressive detection failed: {e}")
            return None

    def remove_duplicate_faces(self, faces, overlap_threshold=0.3):
        """Remove overlapping face detections - EXACTLY same as ingestion."""
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

    def analyze_query_demographics(self, face_crop: np.ndarray) -> dict:
        """
        Analyze demographics of the query face using DeepFace's built-in analysis.

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
                # CRITICAL FIX: Keep original DeepFace format to match stored data
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

            logger.info(f"Query demographics extracted: Gender={demographics['gender']}, Age={demographics.get('age', 'unknown')}, Emotion={demographics['dominant_emotion']}, Race={demographics['dominant_race']}")

        except Exception as e:
            logger.warning(f"Query demographic analysis failed: {e}")
            # Keep default unknown values

        return demographics

def main():
    st.set_page_config(
        page_title="CCTV Reverse Face Search",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 CCTV Reverse Face Search System")
    st.markdown("Upload an image containing a face to search for similar faces in CCTV footage.")

    # Initialize the app
    if 'face_app' not in st.session_state:
        st.session_state.face_app = FaceSearchApp()

    face_app = st.session_state.face_app

    # Check if database is populated
    if not face_app.metadata:
        st.error("⚠️ No face data found in database!")
        st.info("Please run `python ingest.py` first to process a video and populate the database.")
        return

    st.success(f"✅ Database contains {len(face_app.metadata)} face embeddings")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing a face to search for matches"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Query Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert PIL to numpy array for processing
            image_array = np.array(image)

            # Generate embedding
            with st.spinner("Generating embedding for query image..."):
                result = face_app.generate_query_embedding(image_array)

            if result is None:
                st.error("❌ No face detected in the uploaded image. Please upload an image with a clear face.")
                return

            # Unpack the result
            if isinstance(result, tuple) and len(result) == 2:
                query_embedding, query_demographics = result
            else:
                # Fallback for old format
                query_embedding = result
                query_demographics = {'gender': 'unknown', 'age_range': 'unknown', 'dominant_emotion': 'unknown', 'dominant_race': 'unknown'}

            st.success("✅ Face detected and embedding generated!")

            # Display query demographics
            st.subheader("📊 Query Image Analysis")
            st.markdown(f"""
            **Gender:** {query_demographics.get('gender', 'unknown')}
            **Age Range:** {query_demographics.get('age_range', 'unknown')}
            **Emotion:** {query_demographics.get('dominant_emotion', 'unknown')}
            **Race:** {query_demographics.get('dominant_race', 'unknown')}
            """)

        with col2:
            st.subheader("Search Results")

            # Search for similar faces
            with st.spinner("Searching for similar faces..."):
                results = face_app.search_similar_faces(query_embedding, query_demographics, similarity_threshold=0.05, max_results=50)

            # Show detailed similarity debugging
            st.markdown("#### 🔍 Similarity Analysis")
            if results:
                st.info(f"Found {len(results)} faces above 5% similarity threshold")

                # Show top 10 results with detailed scores
                debug_data = []
                for i, result in enumerate(results[:10]):
                    # Use the new processed result format
                    similarity_score = result['similarity_score']
                    adjusted_similarity = result['adjusted_similarity']
                    distance_score = result['distance_score']
                    payload = result['payload']

                    debug_data.append({
                        'Rank': i+1,
                        'Similarity': f"{similarity_score:.4f}",
                        'Adjusted': f"{adjusted_similarity:.4f}",
                        'Distance': f"{distance_score:.4f}",
                        'Quality': f"{payload.get('quality_score', 0.5):.3f}",
                        'Frame': payload['frame_number'],
                        'Timestamp': f"{payload['timestamp']:.1f}s"
                    })

                st.dataframe(debug_data, use_container_width=True)

                best_score = results[0]['similarity_score']
                best_adjusted = results[0]['adjusted_similarity']

                if best_score < 0.3:
                    st.error(f"⚠️ Best match only {best_score:.1%} similar - this indicates embedding inconsistency!")
                    st.markdown("""
                    **Possible causes:**
                    - Different preprocessing between ingestion and query
                    - Model state inconsistency
                    - Face detection/cropping differences
                    """)
                elif best_score < 0.7:
                    st.warning(f"🔍 Best match is {best_score:.1%} similar (adjusted: {best_adjusted:.1%}) - moderate accuracy")
                else:
                    st.success(f"🎯 Best match is {best_score:.1%} similar (adjusted: {best_adjusted:.1%}) - good accuracy!")
            else:
                st.error("❌ No faces found even at 5% threshold!")
                st.markdown("""
                **This suggests a fundamental embedding problem. Try:**
                1. Re-running ingestion: `python ingest.py`
                2. Using a different query image
                3. Checking if the face is clearly visible
                """)

            if not results:
                return

            # Display results (now properly sorted by similarity)
            for i, result in enumerate(results):
                # Use the new processed result format
                similarity_score = result['similarity_score']
                adjusted_similarity = result['adjusted_similarity']
                payload = result['payload']

                st.markdown(f"### Match #{i+1}")

                # Create columns for result display
                result_col1, result_col2 = st.columns([1, 2])

                with result_col1:
                    # Load and display face image
                    face_image = face_app.load_face_image(payload['face_path'])
                    if face_image:
                        st.image(face_image, caption=f"Match {i+1}", use_column_width=True)
                    else:
                        st.warning("Face image not found")

                with result_col2:
                    # Display metadata with both similarity scores - NO NESTED COLUMNS
                    st.metric("Similarity Score", f"{similarity_score:.3f}")
                    st.metric("Adjusted Score", f"{adjusted_similarity:.3f}")

                    # Show quality information
                    quality_score = payload.get('quality_score', 0.5)
                    detection_method = payload.get('detection_backend', 'unknown')
                    embedding_model = payload.get('embedding_model', 'unknown')

                    # Display demographic information if available
                    gender = payload.get('gender', 'unknown')
                    age_range = payload.get('age_range', 'unknown')
                    emotion = payload.get('dominant_emotion', 'unknown')
                    race = payload.get('dominant_race', 'unknown')

                    st.markdown(f"""
                    **Video:** {payload['video_name']}
                    **Frame:** {payload['frame_number']}
                    **Camera:** {payload['camera_id']}
                    **Timestamp:** {payload['timestamp']:.1f}s
                    **Quality:** {quality_score:.3f} | **Detection:** {detection_method} | **Model:** {embedding_model}

                    **Demographics:**
                    **Gender:** {gender} | **Age:** {age_range} | **Emotion:** {emotion} | **Race:** {race}
                    """)

                    # Similarity indicator based on adjusted score
                    if adjusted_similarity > 0.8:
                        st.success("🎯 High confidence match")
                    elif adjusted_similarity > 0.6:
                        st.warning("🔍 Medium confidence match")
                    else:
                        st.info("💭 Low confidence match")

                st.divider()

    # Sidebar with information
    with st.sidebar:
        st.header("System Information")
        st.info(f"**Database Status:** {len(face_app.metadata)} faces indexed")

        st.header("How it works")
        st.markdown("""
        1. **Upload** an image containing a face
        2. **Face Detection** using RetinaFace
        3. **Embedding Generation** using ArcFace model
        4. **Vector Search** in Qdrant database
        5. **Results** ranked by similarity score
        """)

        st.header("Tips")
        st.markdown("""
        - Use clear, well-lit face images
        - Front-facing faces work best
        - Higher similarity scores indicate better matches
        - Scores above 0.8 are typically reliable matches
        """)

if __name__ == "__main__":
    main()
