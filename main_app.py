#!/usr/bin/env python3
"""
Professional Face Search Application
Using proper similarity thresholds and normalized embeddings.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import logging

from face_recognition_engine import ProfessionalFaceRecognitionEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfessionalFaceSearchApp:
    def __init__(self):
        """Initialize the professional face search application."""
        self.engine = ProfessionalFaceRecognitionEngine()
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

    def search_face(self, image: np.ndarray, threshold_level: str = 'normal') -> tuple:
        """
        Search for faces using the professional engine.

        Args:
            image: Input image as numpy array
            threshold_level: Threshold level to use

        Returns:
            Tuple of (results, query_metadata) or (None, None)
        """
        try:
            # Generate embedding using professional engine
            result = self.engine.generate_embedding(image, model_name='ArcFace')

            if result is None:
                return None, None

            embedding, query_metadata = result

            # Search for similar faces
            search_results = self.engine.search_faces(
                query_embedding=embedding,
                query_metadata=query_metadata,
                threshold_level=threshold_level,
                max_results=20
            )

            return search_results, query_metadata

        except Exception as e:
            logger.error(f"Error in face search: {e}")
            return None, None

    def load_face_image(self, face_path: str):
        """Load and return face image from path."""
        if os.path.exists(face_path):
            return Image.open(face_path)
        return None


def main():
    st.set_page_config(
        page_title="Professional CCTV Face Search",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Professional CCTV Face Search System")
    st.markdown("""
    **Advanced Face Recognition with Proper Similarity Thresholds**

    This system uses professional-grade similarity calculation with:
    - L2-normalized embeddings for accurate cosine similarity
    - Strict similarity thresholds (0.35-0.65 range)
    - Enhanced demographic filtering
    - Quality-based result ranking
    """)

    # Initialize the app
    if 'face_app' not in st.session_state:
        st.session_state.face_app = ProfessionalFaceSearchApp()

    face_app = st.session_state.face_app

    # Check if database is populated
    if not face_app.metadata:
        st.error("⚠️ No face data found in database!")
        st.info("Please run `python ingest.py` first to process a video and populate the database.")
        return

    st.success(f"✅ Database contains {len(face_app.metadata)} face embeddings")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Search Configuration")

        # Threshold selection
        threshold_level = st.selectbox(
            "Similarity Threshold Level",
            options=['strict', 'normal', 'loose', 'minimum'],
            index=1,  # Default to 'normal'
            help="""
            - **Strict (0.65)**: Very high confidence matches only
            - **Normal (0.55)**: Balanced approach (recommended)
            - **Loose (0.45)**: More permissive matching
            - **Minimum (0.35)**: Very permissive (use with caution)
            """
        )

        # Show threshold info
        threshold_info = face_app.engine.get_threshold_info()
        current_threshold = threshold_info['thresholds'][threshold_level]
        st.info(f"**Current Threshold**: {current_threshold:.2f}")
        st.caption(threshold_info['recommendations'][threshold_level])

        st.header("📊 System Information")
        st.info(f"**Database Status**: {len(face_app.metadata)} faces indexed")

        st.header("🎯 How it works")
        st.markdown("""
        1. **Upload** an image with a face
        2. **L2 Normalization** of face embeddings
        3. **Cosine Similarity** calculation
        4. **Strict Thresholding** to prevent false matches
        5. **Demographic Filtering** for accuracy
        6. **Results** ranked by adjusted similarity
        """)

    # Main content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Query Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing a clear face"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Query Image", use_column_width=True)

            # Convert PIL to numpy array
            image_array = np.array(image)

            # Search for faces
            with st.spinner("🔍 Analyzing face and searching database..."):
                search_results, query_metadata = face_app.search_face(
                    image_array,
                    threshold_level=threshold_level
                )

            if search_results is None:
                st.error("❌ No face detected in the uploaded image. Please upload an image with a clear, visible face.")
            else:
                # Display query analysis
                st.subheader("📊 Query Analysis")
                if query_metadata and 'demographics' in query_metadata:
                    demographics = query_metadata['demographics']
                    quality_score = query_metadata.get('quality_score', 0.0)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Gender", demographics.get('gender', 'Unknown'))
                        st.metric("Age Range", demographics.get('age_range', 'Unknown'))
                    with col_b:
                        st.metric("Emotion", demographics.get('dominant_emotion', 'Unknown'))
                        st.metric("Quality Score", f"{quality_score:.2f}")

    with col2:
        if uploaded_file is not None and search_results is not None:
            st.subheader("🎯 Search Results")

            if not search_results:
                st.warning(f"""
                ⚠️ **No matches found above {current_threshold:.2f} similarity threshold**

                This is actually **good news**! It means the system is working correctly
                and preventing false matches.

                **Why no matches?**
                - Your query face is genuinely different from faces in the database
                - The strict thresholds are preventing false positives
                - This is the expected behavior for faces not in the database

                **Try:**
                - Using a face from the `data/preview_faces/` folder
                - Lowering the threshold level (but expect more false positives)
                """)

                # Show near-misses for debugging
                st.subheader("🔍 Debug: Near-Misses")
                st.caption("These are the closest matches (below threshold):")

                # Get results with minimum threshold for debugging
                debug_results, _ = face_app.search_face(image_array, threshold_level='minimum')
                if debug_results:
                    for i, result in enumerate(debug_results[:3]):
                        payload = result['payload']
                        similarity = result['similarity']

                        debug_col1, debug_col2 = st.columns([1, 2])
                        with debug_col1:
                            face_image = face_app.load_face_image(payload.get('face_path', ''))
                            if face_image:
                                # Apply transformations: invert and rotate 90 degrees right
                                face_array = np.array(face_image)
                                face_image_inverted = 255 - face_array
                                face_image_rotated = np.rot90(face_image_inverted, k=-1)
                                st.image(face_image_rotated, width=100)
                        with debug_col2:
                            st.caption(f"""
                            **Similarity**: {similarity:.3f} (below {current_threshold:.2f} threshold)
                            **Gender**: {payload.get('gender', 'Unknown')}
                            **Quality**: {payload.get('quality_score', 0):.2f}
                            **Frame**: {payload.get('frame_number', 'Unknown')}
                            """)
            else:
                # Display results
                st.success(f"✅ Found {len(search_results)} high-confidence matches!")

                # Results summary
                avg_similarity = np.mean([r['similarity'] for r in search_results])
                max_similarity = max([r['similarity'] for r in search_results])

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Total Matches", len(search_results))
                with metric_col2:
                    st.metric("Avg Similarity", f"{avg_similarity:.2f}")
                with metric_col3:
                    st.metric("Best Match", f"{max_similarity:.2f}")

                # Display individual results
                for i, result in enumerate(search_results):
                    payload = result['payload']
                    similarity = result['similarity']
                    adjusted_similarity = result['adjusted_similarity']

                    # Create expandable result
                    with st.expander(f"Match {i+1}: {similarity:.2f} similarity", expanded=(i < 3)):
                        result_col1, result_col2 = st.columns([1, 2])

                        with result_col1:
                            # Load and display face image
                            face_image = face_app.load_face_image(payload.get('face_path', ''))
                            if face_image:
                                # Apply transformations: invert and rotate 90 degrees right
                                face_array = np.array(face_image)
                                face_image_inverted = 255 - face_array
                                face_image_rotated = np.rot90(face_image_inverted, k=-1)
                                st.image(face_image_rotated, width=150)
                            else:
                                st.error("Image not found")

                        with result_col2:
                            st.markdown(f"""
                            **Similarity Score**: {similarity:.3f}
                            **Adjusted Score**: {adjusted_similarity:.3f}
                            **Quality Score**: {payload.get('quality_score', 0):.2f}

                            **Demographics**:
                            - Gender: {payload.get('gender', 'Unknown')}
                            - Age Range: {payload.get('age_range', 'Unknown')}
                            - Emotion: {payload.get('dominant_emotion', 'Unknown')}
                            - Race: {payload.get('dominant_race', 'Unknown')}

                            **Technical Details**:
                            - Frame: {payload.get('frame_number', 'Unknown')}
                            - Timestamp: {payload.get('timestamp', 'Unknown')}s
                            - Detection: {payload.get('detection_backend', 'Unknown')}
                            - Model: {payload.get('embedding_model', 'Unknown')}
                            """)

                # Show similarity distribution
                if len(search_results) > 1:
                    st.subheader("📈 Similarity Distribution")
                    similarities = [r['similarity'] for r in search_results]
                    st.bar_chart(similarities)


if __name__ == "__main__":
    main()
