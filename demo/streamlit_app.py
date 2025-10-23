
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

st.set_page_config(
    page_title="Facial Authentication Demo",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Enterprise Facial Authentication System")
st.markdown("---")

@st.cache_resource
def load_models():
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

st.sidebar.header("Settings")
mode = st.sidebar.selectbox(
    "Select Mode",
    ["🎥 Real-Time Camera", "Face Detection Demo", "Liveness Check", "Embedding Extraction", "System Info"]
)

input_source = st.sidebar.radio(
    "Input Source",
    ["Camera", "Upload Image"],
    index=0 if mode == "🎥 Real-Time Camera" else 1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
)

if mode == "🎥 Real-Time Camera":
    st.header("🎥 Real-Time Camera Authentication")

    st.info("📷 Use your webcam for real-time face detection and authentication")

    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        show_landmarks = st.checkbox("Show Landmarks", value=True)
    with col_opt2:
        auto_analyze = st.checkbox("Auto Analyze", value=True)

    camera_image = st.camera_input("Take a picture", key="camera_realtime")

    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Captured Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

        with col2:
            st.subheader("🔍 Analysis")

            with st.spinner("Analyzing..."):
                start_time = time.time()

                detections = models['detector'].detect(image)

                if len(detections) == 0:
                    st.warning("⚠️ No face detected. Please try again.")
                else:
                    detection = detections[0]

                    result_image = image.copy()
                    bbox = detection.bbox.astype(int)
                    cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                    for name, (x, y) in detection.landmarks.items():
                        cv2.circle(result_image, (int(x), int(y)), 3, (0, 0, 255), -1)

                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                            caption='Detection Result', use_column_width=True)

                    st.metric("Face Confidence", f"{detection.confidence:.1%}")
                    st.metric("Quality Score", f"{detection.quality_score:.1%}")

                    st.markdown("---")
                    st.subheader("🎭 Liveness Check")

                    face_region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    is_live, liveness_score = models['liveness'].predict(face_region)

                    if is_live:
                        st.success(f"✅ LIVE PERSON ({liveness_score:.1%})")
                    else:
                        st.error(f"⚠️ SPOOF DETECTED ({liveness_score:.1%})")

                    st.progress(liveness_score)

                    elapsed = (time.time() - start_time) * 1000
                    st.metric("Processing Time", f"{elapsed:.0f} ms")

                    if st.button("🧬 Extract Embeddings"):
                        with st.spinner("Extracting embeddings..."):
                            face_160 = models['aligner'].align(image, detection, 160)
                            face_224 = models['aligner'].align(image, detection, 224)
                            embeddings = models['extractor'].extract_all_embeddings(face_160, face_224)

                            st.success(f"✅ Extracted {len(embeddings)} embeddings")
                            for model_name, embedding in embeddings.items():
                                st.write(f"**{model_name.upper()}**: {embedding.shape[0]}D vector (L2 norm: {np.linalg.norm(embedding):.3f})")

elif mode == "Face Detection Demo":
    st.header("📸 Face Detection Demo")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")

        image = None
        if input_source == "Camera":
            camera_image = st.camera_input("Take a picture")
            if camera_image is not None:
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:

            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)

            if st.button("Detect Faces"):
                with st.spinner("Detecting faces..."):
                    start_time = time.time()
                    detections = models['detector'].detect(image)
                    elapsed = (time.time() - start_time) * 1000

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
        if image is not None and 'detections' in locals():
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

    st.info("Check if it's a live person or a spoof (photo/video/mask)")

    image = None
    if input_source == "Camera":
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption='Input Image', use_column_width=True)

        if st.button("Check Liveness"):
            with st.spinner("Analyzing..."):
                detection = models['detector'].detect_largest(image)

                if detection is None:
                    st.error("❌ No face detected!")
                else:
                    bbox = detection.bbox.astype(int)
                    face_region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    is_live, liveness_score = models['liveness'].predict(face_region)

                    with col2:
                        if is_live:
                            st.success(f"✅ LIVE PERSON DETECTED")
                            st.metric("Liveness Score", f"{liveness_score:.3f}", delta="Pass")
                        else:
                            st.error(f"⚠️ SPOOF DETECTED")
                            st.metric("Liveness Score", f"{liveness_score:.3f}", delta="Fail", delta_color="inverse")

                        st.progress(liveness_score)

                        st.write("**Texture Analysis:**")
                        texture = models['liveness'].texture_analysis(face_region)
                        st.json(texture)

elif mode == "Embedding Extraction":
    st.header("🧬 Face Embedding Extraction")

    st.info("Extract high-dimensional face embeddings using multiple models")

    image = None
    if input_source == "Camera":
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption='Input Image', use_column_width=True)

        if st.button("Extract Embeddings"):
            with st.spinner("Processing..."):
                detection = models['detector'].detect_largest(image)

                if detection is None:
                    st.error("❌ No face detected!")
                else:
                    face_160 = models['aligner'].align(image, detection, 160)
                    face_224 = models['aligner'].align(image, detection, 224)

                    embeddings = models['extractor'].extract_all_embeddings(face_160, face_224)

                    with col2:
                        st.success("✅ Embeddings extracted successfully!")

                        st.write("**Embedding Statistics:**")

                        for model_name, embedding in embeddings.items():
                            with st.expander(f"{model_name.upper()} Embedding"):
                                st.write(f"Shape: {embedding.shape}")
                                st.write(f"Mean: {np.mean(embedding):.4f}")
                                st.write(f"Std: {np.std(embedding):.4f}")
                                st.write(f"L2 Norm: {np.linalg.norm(embedding):.4f}")

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

st.markdown("---")
st.markdown(