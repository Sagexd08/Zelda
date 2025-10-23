"""
Streamlit Demo Application
Interactive webcam-based facial authentication
"""

import sys
sys.path.append('..')

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path

from app.models.face_detector import get_face_detector
from app.models.face_aligner import get_face_aligner
from app.models.embedding_extractor import get_embedding_extractor, cosine_similarity
from app.models.fusion_model import get_fusion_model
from app.models.liveness_detector import get_liveness_detector
from app.core.config import settings

# Page config
st.set_page_config(
    page_title="Facial Authentication Demo",
    page_icon="🔐",
    layout="wide"
)

# Title
st.title("🔐 Enterprise Facial Authentication System")
st.markdown("---")

# Initialize models (cached)
@st.cache_resource
def load_models():
    """Load all ML models (cached)"""
    return {
        'detector': get_face_detector(),
        'aligner': get_face_aligner(),
        'extractor': get_embedding_extractor(),
        'fusion': get_fusion_model(),
        'liveness': get_liveness_detector()
    }

with st.spinner("Loading models..."):
    models = load_models()
    st.success("✓ Models loaded successfully!")

# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Face Detection Demo", "Liveness Check", "Embedding Extraction", "System Info"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **Enterprise Facial Authentication System**
    
    Features:
    - Multi-model face recognition
    - Advanced liveness detection
    - Real-time processing
    - Secure encryption
    """
)

# Main content based on mode
if mode == "Face Detection Demo":
    st.header("📸 Face Detection Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)
            
            # Detect faces
            if st.button("Detect Faces"):
                with st.spinner("Detecting faces..."):
                    start_time = time.time()
                    detections = models['detector'].detect(image)
                    elapsed = (time.time() - start_time) * 1000
                    
                    # Draw results
                    result_image = image.copy()
                    for detection in detections:
                        bbox = detection.bbox.astype(int)
                        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        
                        for name, (x, y) in detection.landmarks.items():
                            cv2.circle(result_image, (int(x), int(y)), 3, (0, 0, 255), -1)
                        
                        cv2.putText(result_image, f"{detection.confidence:.2f}", 
                                  (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 2)
    
    with col2:
        if uploaded_file is not None and 'detections' in locals():
            st.subheader("Detection Results")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                    caption='Detected Faces', use_column_width=True)
            
            st.metric("Faces Detected", len(detections))
            st.metric("Processing Time", f"{elapsed:.1f} ms")
            
            if detections:
                st.write("**Detection Details:**")
                for i, det in enumerate(detections):
                    with st.expander(f"Face {i+1}"):
                        st.write(f"Confidence: {det.confidence:.3f}")
                        st.write(f"Quality Score: {det.quality_score:.3f}")
                        st.write(f"Bbox: {det.bbox.astype(int).tolist()}")

elif mode == "Liveness Check":
    st.header("🎭 Liveness Detection")
    
    st.info("Upload a face image to check if it's a live person or a spoof (photo/video/mask)")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption='Input Image', use_column_width=True)
        
        if st.button("Check Liveness"):
            with st.spinner("Analyzing..."):
                # Detect face
                detection = models['detector'].detect_largest(image)
                
                if detection is None:
                    st.error("❌ No face detected!")
                else:
                    # Extract face region
                    bbox = detection.bbox.astype(int)
                    face_region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    
                    # Check liveness
                    is_live, liveness_score = models['liveness'].predict(face_region)
                    
                    with col2:
                        if is_live:
                            st.success(f"✅ LIVE PERSON DETECTED")
                            st.metric("Liveness Score", f"{liveness_score:.3f}", delta="Pass")
                        else:
                            st.error(f"⚠️ SPOOF DETECTED")
                            st.metric("Liveness Score", f"{liveness_score:.3f}", delta="Fail", delta_color="inverse")
                        
                        # Show progress bar
                        st.progress(liveness_score)
                        
                        # Texture analysis
                        st.write("**Texture Analysis:**")
                        texture = models['liveness'].texture_analysis(face_region)
                        st.json(texture)

elif mode == "Embedding Extraction":
    st.header("🧬 Face Embedding Extraction")
    
    st.info("Extract high-dimensional face embeddings using multiple models")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption='Input Image', use_column_width=True)
        
        if st.button("Extract Embeddings"):
            with st.spinner("Processing..."):
                # Detect and align
                detection = models['detector'].detect_largest(image)
                
                if detection is None:
                    st.error("❌ No face detected!")
                else:
                    face_160 = models['aligner'].align(image, detection, 160)
                    face_224 = models['aligner'].align(image, detection, 224)
                    
                    # Extract embeddings
                    embeddings = models['extractor'].extract_all_embeddings(face_160, face_224)
                    
                    with col2:
                        st.success("✅ Embeddings extracted successfully!")
                        
                        # Show embedding statistics
                        st.write("**Embedding Statistics:**")
                        
                        for model_name, embedding in embeddings.items():
                            with st.expander(f"{model_name.upper()} Embedding"):
                                st.write(f"Shape: {embedding.shape}")
                                st.write(f"Mean: {np.mean(embedding):.4f}")
                                st.write(f"Std: {np.std(embedding):.4f}")
                                st.write(f"L2 Norm: {np.linalg.norm(embedding):.4f}")
                                
                                # Visualize embedding
                                st.line_chart(embedding[:100])

elif mode == "System Info":
    st.header("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        st.json({
            "Version": settings.APP_VERSION,
            "Environment": settings.ENVIRONMENT,
            "Device": str(settings.get_device()),
            "Face Size": settings.FACE_SIZE,
            "Verification Threshold": settings.VERIFICATION_THRESHOLD,
            "Liveness Threshold": settings.LIVENESS_THRESHOLD
        })
    
    with col2:
        st.subheader("Features")
        features = {
            "Face Detection": "✅ RetinaFace + MTCNN",
            "Liveness Detection": "✅ CNN + Temporal",
            "Depth Estimation": "✅" if settings.ENABLE_DEPTH_ESTIMATION else "❌",
            "Adaptive Learning": "✅" if settings.ENABLE_ONLINE_LEARNING else "❌",
            "Challenge-Response": "✅" if settings.ENABLE_CHALLENGE_RESPONSE else "❌",
            "Voice Auth": "✅" if settings.ENABLE_VOICE_AUTH else "❌"
        }
        
        for feature, status in features.items():
            st.write(f"**{feature}:** {status}")
    
    st.markdown("---")
    
    st.subheader("Model Information")
    
    models_info = {
        "Model": ["ArcFace", "FaceNet", "MobileFaceNet", "Liveness", "Fusion"],
        "Type": ["ResNet100", "Inception-ResNet-v1", "MobileNet-v2", "ResNet18", "MLP"],
        "Size": ["249 MB", "107 MB", "4 MB", "45 MB", "<1 MB"],
        "Accuracy": ["99.8%", "99.6%", "99.2%", "98.5%", "+3%"]
    }
    
    st.table(models_info)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Enterprise-Grade Facial Authentication System v1.0.0</p>
        <p>Built with FastAPI, PyTorch, and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

