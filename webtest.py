import cv2
import numpy as np
import streamlit as st
from datetime import datetime

# WebRTC bits
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# -----------------------------
# App config & styling
# -----------------------------
st.set_page_config(page_title="Face Recognition App", page_icon="🎭", layout="wide")
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, #4b0000, #800020); /* wine gradient */
            color: white;
        }
        h1, h2, h3, h4 {
            color: #f1c40f;
            text-align: center;
        }
        .stButton>button {
            background-color: #27ae60;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #2ecc71;
        }
        .block-container { padding-top: 1rem; }
        .caption { font-size: 0.9rem; opacity: 0.9; }
        .panel {
            background: rgba(0,0,0,0.25);
            border-radius: 16px;
            padding: 1rem 1.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load models (globals)
# -----------------------------
# LBPH recognizer (requires opencv-contrib-python)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")

# Class labels (update as per your training)
face_labels = {0: "deepak", 1: "pratham"}

# Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Recognition helper
def recognize_face(gray_frame, x, y, w, h, conf_thresh=100):
    face = gray_frame[y:y + h, x:x + w]
    label, confidence = face_recognizer.predict(face)
    if confidence < conf_thresh:
        return face_labels.get(label, "Unknown"), confidence
    return "Unknown", confidence

# -----------------------------
# Title
# -----------------------------
st.title("🎭 Face Recognition App")
st.markdown('<div class="caption" style="text-align:center;">Upload an image or use real-time WebRTC video to recognize faces.</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Choose Mode:", ["📤 Upload Image", "🌐 WebRTC (Live)"])
conf_threshold = st.sidebar.slider("Recognition Confidence Threshold (lower = stricter)", 10, 150, 100, 1)

# -----------------------------
# Mode 1: Upload Image
# -----------------------------
if mode == "📤 Upload Image":
    st.subheader("📤 Upload an Image for Recognition")
    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            detections = []
            for (x, y, w, h) in faces:
                name, conf = recognize_face(gray, x, y, w, h, conf_thresh=conf_threshold)
                detections.append((name, conf, (x, y, w, h)))
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="🖼️ Processed Image")

            if detections:
                st.markdown("**Detections:**")
                for name, conf, _ in detections:
                    st.write(f"- {name}  (confidence: {conf:.2f})")
            else:
                st.info("No faces detected.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Mode 2: WebRTC (Live)
# -----------------------------
elif mode == "🌐 WebRTC (Live)":
    st.subheader("🌐 Real-time Face Recognition via WebRTC")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write(
        "Grant camera permission in your browser. This works across devices (laptop/phone). "
        "Use your browser’s camera selector to choose a device (DroidCam appears as a normal camera locally)."
    )

    # Snapshot trigger flag in session_state (clicked only once)
    if "take_snapshot" not in st.session_state:
        st.session_state.take_snapshot = False

    col1, col2 = st.columns([1, 1])
    with col1:
        snapshot_click = st.button("📸 Capture Snapshot")

    with col2:
        st.caption("Snapshots save frames with bounding boxes to the app folder.")

    # If snapshot button clicked, set the flag (one-time)
    if snapshot_click:
        st.session_state.take_snapshot = True

    # WebRTC config
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Transformer class for real-time processing
    class FaceRecTransformer(VideoTransformerBase):
        def __init__(self):
            self.last_frame = None  # store last processed BGR frame for snapshots

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                name, conf = recognize_face(gray, x, y, w, h, conf_thresh=conf_threshold)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            self.last_frame = img  # keep latest processed frame

            # If a snapshot was requested, save exactly once here then reset flag
            if st.session_state.get("take_snapshot", False) and self.last_frame is not None:
                filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, self.last_frame)
                st.toast(f"✅ Snapshot saved: {filename}")
                st.session_state.take_snapshot = False  # reset so it doesn't save every frame

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Start WebRTC streamer
    ctx = webrtc_streamer(
        key="face-recognition-webrtc",
        mode="recvonly",  # we only receive video from the client
        video_transformer_factory=FaceRecTransformer,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        rtc_configuration=rtc_config,
    )

    # Info & tips
    st.markdown("---")
    st.caption(
        "If you don’t see the camera: ensure browser permissions are granted, "
        "and check the camera dropdown in the browser (DroidCam will appear there if installed locally)."
    )
    st.markdown('</div>', unsafe_allow_html=True)
