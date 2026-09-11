import cv2
import av
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import time
import threading

# ============================================================
# FRIDGEGUARD AI — REFRIGERATOR QUALITY INSPECTION SYSTEM
# ============================================================

st.set_page_config(
    page_title="FridgeGuard AI | Refrigerator Quality Inspection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME SYSTEM
# ============================================================
THEMES = {
    "Industrial Dark": {
        "bg": "#0B1220", "sidebar": "#101D31", "panel": "#111C2E",
        "panel2": "#16253A", "border": "#263B57", "primary": "#2D8CFF",
        "primary_hover": "#1E70D6", "text": "#F4F7FB", "muted": "#9BAEC8",
        "success": "#27C281", "danger": "#F05262", "warning": "#F4B740"
    },
    "Professional Light": {
        "bg": "#F4F7FB", "sidebar": "#FFFFFF", "panel": "#FFFFFF",
        "panel2": "#EDF3FA", "border": "#D7E0EA", "primary": "#1769D1",
        "primary_hover": "#1256AB", "text": "#172033", "muted": "#64748B",
        "success": "#159B63", "danger": "#D83A4A", "warning": "#C88700"
    }
}

if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Industrial Dark"


def apply_theme(theme):
    st.markdown(f"""
    <style>
    .stApp {{ background: {theme['bg']}; color: {theme['text']}; }}
    section[data-testid="stSidebar"] {{
        background: {theme['sidebar']};
        border-right: 1px solid {theme['border']};
    }}
    h1,h2,h3,h4,h5,p,label {{ color: {theme['text']} !important; }}

    .app-header {{
        background: {theme['panel']};
        border: 1px solid {theme['border']};
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 18px;
    }}
    .header-title {{
        color: {theme['text']}; font-size: 2.15rem;
        font-weight: 750; line-height: 1.15;
    }}
    .header-subtitle {{ color: {theme['muted']}; margin-top: 6px; }}

    .system-badge {{
        background: rgba(39,194,129,.12);
        border: 1px solid rgba(39,194,129,.35);
        color: {theme['success']};
        border-radius: 999px; padding: 8px 14px;
        font-weight: 700; text-align: center;
    }}

    .kpi-card {{
        background: {theme['panel']};
        border: 1px solid {theme['border']};
        border-radius: 14px; padding: 16px 18px;
        min-height: 102px;
    }}
    .kpi-label {{
        color: {theme['muted']}; font-size: .78rem;
        text-transform: uppercase; letter-spacing: .08em;
    }}
    .kpi-value {{
        color: {theme['text']}; font-size: 1.75rem;
        font-weight: 750; margin-top: 7px;
    }}

    .status-pass {{
        background: rgba(39,194,129,.12);
        border: 1px solid rgba(39,194,129,.4);
        color: {theme['success']}; border-radius: 14px;
        padding: 20px; text-align: center; font-weight: 750;
        font-size: 1.2rem;
    }}
    .status-fail {{
        background: rgba(240,82,98,.12);
        border: 1px solid rgba(240,82,98,.4);
        color: {theme['danger']}; border-radius: 14px;
        padding: 20px; text-align: center; font-weight: 750;
        font-size: 1.2rem;
    }}
    .status-waiting {{
        background: {theme['panel2']}; border: 1px solid {theme['border']};
        color: {theme['muted']}; border-radius: 14px;
        padding: 20px; text-align: center;
    }}

    .stButton > button {{
        background: {theme['primary']}; color: white;
        border: none; border-radius: 10px;
        font-weight: 650; min-height: 42px;
    }}
    .stButton > button:hover {{ background: {theme['primary_hover']}; color: white; }}

    [data-testid="stMetric"] {{
        background: {theme['panel']}; border: 1px solid {theme['border']};
        border-radius: 12px; padding: 14px;
    }}

    button[data-baseweb="tab"] {{ font-weight: 650; }}
    #MainMenu {{visibility:hidden;}}
    footer {{visibility:hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Timestamp", "Status", "glass_shelves", "door_balcony",
        "fruit_box", "vegetable_box", "egg_tray", "ice_tray_pack"
    ])

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    try:
        st.image("Screenshot 2026-08-18 163605.png", use_container_width=True)
    except Exception:
        st.markdown("## FridgeGuard AI")

    st.markdown("---")
    st.markdown("### Appearance")
    selected_theme = st.selectbox(
        "Dashboard Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_name)
    )
    st.session_state.theme_name = selected_theme

apply_theme(THEMES[st.session_state.theme_name])

with st.sidebar:
    st.markdown("---")
    st.markdown("### Vision Parameters")
    st.caption("Advanced AI detection settings")

    with st.expander("⚙ Detection Settings", expanded=False):
        conf_threshold = st.slider(
            "Confidence Threshold", 0.10, 1.00, 0.25, 0.05,
            help="Minimum confidence required for a detection."
        )
        iou_threshold = st.slider(
            "IoU (Overlap) Threshold", 0.10, 1.00, 0.45, 0.05,
            help="Controls suppression of overlapping boxes."
        )

    st.markdown("---")
    st.markdown('<div class="system-badge">● SYSTEM READY</div>', unsafe_allow_html=True)
    st.caption("YOLO-powered visual assembly verification")

# Defaults remain available even if expander is closed
if "conf_threshold" not in locals():
    conf_threshold = 0.25
    iou_threshold = 0.45

# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner="Loading AI inspection model...")
def load_model():
    return YOLO("best.pt")

model = load_model()
MODEL_LOCK = threading.Lock()

REQUIRED_PARTS = {
    "glass_shelves": 6,
    "door_balcony": 7,
    "fruit_box": 1,
    "vegetable_box": 1,
    "egg_tray": 1,
    "ice_tray_pack": 1,
}

# ============================================================
# ML FUNCTIONS
# ============================================================
def analyze_image(image_array, conf_thresh, iou_thresh):
    start = time.perf_counter()
    # The live camera callback and image-upload workflow can run at the same time.
    # A lock prevents concurrent access to the shared YOLO model.
    with MODEL_LOCK:
        results = model.predict(
            source=image_array,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False
        )
    inference_ms = (time.perf_counter() - start) * 1000

    result = results[0]
    annotated = cv2.cvtColor(result.plot(line_width=3), cv2.COLOR_BGR2RGB)

    counts = {part: 0 for part in REQUIRED_PARTS}
    confidences = {part: [] for part in REQUIRED_PARTS}

    for box in result.boxes:
        cls_name = model.names[int(box.cls[0].item())]
        confidence = float(box.conf[0].item())
        if cls_name in counts:
            counts[cls_name] += 1
            confidences[cls_name].append(confidence)

    missing = {
        part: max(required - counts[part], 0)
        for part, required in REQUIRED_PARTS.items()
    }
    extra = {
        part: max(counts[part] - required, 0)
        for part, required in REQUIRED_PARTS.items()
    }

    total_expected = sum(REQUIRED_PARTS.values())
    total_detected = sum(counts.values())
    total_verified = sum(min(counts[p], REQUIRED_PARTS[p]) for p in REQUIRED_PARTS)

    all_conf = [c for values in confidences.values() for c in values]
    avg_conf = float(np.mean(all_conf)) if all_conf else 0.0

    is_passed = all(value == 0 for value in missing.values()) and all(value == 0 for value in extra.values())

    return {
        "annotated": annotated,
        "counts": counts,
        "confidences": confidences,
        "missing": missing,
        "extra": extra,
        "is_passed": is_passed,
        "total_expected": total_expected,
        "total_detected": total_detected,
        "total_verified": total_verified,
        "avg_confidence": avg_conf,
        "inference_ms": inference_ms,
    }


class InspectionVideoProcessor(VideoProcessorBase):
    """Runs FridgeGuard AI inference on each browser-camera frame."""

    def __init__(self, conf_thresh, iou_thresh):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.latest_result = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # WebRTC frames arrive as BGR images; the dashboard model pipeline uses RGB.
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.latest_result = analyze_image(
            frame_rgb, self.conf_thresh, self.iou_thresh
        )
        annotated_bgr = cv2.cvtColor(
            self.latest_result["annotated"], cv2.COLOR_RGB2BGR
        )
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


def build_report_dataframe(result):
    rows = []
    for part, required in REQUIRED_PARTS.items():
        found = result["counts"][part]
        confs = result["confidences"][part]
        avg_conf = f"{np.mean(confs) * 100:.1f}%" if confs else "—"

        if found >= required:
            status = "✓ VERIFIED"
        else:
            status = f"✕ MISSING ({required - found})"

        rows.append({
            "Component": part.replace("_", " ").title(),
            "Detected": found,
            "Expected": required,
            "Avg Confidence": avg_conf,
            "Status": status,
            "_class_name": part,
        })
    return pd.DataFrame(rows)


def filtered_class_image(image_array, selected_classes, conf_thresh, iou_thresh):
    """Re-run YOLO only for classes selected in the results table."""
    class_map = {name: class_id for class_id, name in model.names.items()}
    filter_ids = [class_map[name] for name in selected_classes if name in class_map]

    if not filter_ids:
        return None

    filtered_results = model.predict(
        source=image_array,
        conf=conf_thresh,
        iou=iou_thresh,
        classes=filter_ids,
        verbose=False
    )

    return cv2.cvtColor(
        filtered_results[0].plot(line_width=3),
        cv2.COLOR_BGR2RGB
    )


def save_to_history(result):
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Status": "PASS" if result["is_passed"] else "FAIL",
    }
    record.update(result["counts"])
    st.session_state.history = pd.concat(
        [pd.DataFrame([record]), st.session_state.history],
        ignore_index=True
    )


def kpi_card(label, value):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True
    )


def status_card(result=None):
    if result is None:
        st.markdown(
            '<div class="status-waiting"><b>WAITING FOR INSPECTION</b><br>'
            '<span>Start the camera or upload an image.</span></div>',
            unsafe_allow_html=True
        )
    elif result["is_passed"]:
        st.markdown(
            '<div class="status-pass">✓ ASSEMBLY PASSED<br>'
            '<span style="font-size:.9rem;font-weight:400">'
            'All required components verified</span></div>',
            unsafe_allow_html=True
        )
    else:
        missing_names = [
            part.replace("_", " ").title()
            for part, amount in result["missing"].items() if amount > 0
        ]
        st.markdown(
            '<div class="status-fail">✕ ASSEMBLY FAILED<br>'
            f'<span style="font-size:.85rem;font-weight:400">Missing: {", ".join(missing_names)}</span>'
            '</div>',
            unsafe_allow_html=True
        )

# ============================================================
# HEADER
# ============================================================
left, right = st.columns([5, 1])
with left:
    st.markdown(
        '<div class="app-header">'
        '<div class="header-title">FridgeGuard AI</div>'
        '<div class="header-subtitle">Automated Visual Assembly Verification & Quality Control</div>'
        '</div>',
        unsafe_allow_html=True
    )
with right:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="system-badge">● ONLINE</div>', unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab_live, tab_static, tab_analytics = st.tabs([
    "Live Inspection", "Image Inspection", "Production Analytics"
])

# ============================================================
# LIVE INSPECTION
# ============================================================
with tab_live:
    st.markdown("### Live AI Inspection")
    col_video, col_diag = st.columns([2.2, 1], gap="large")

    with col_video:
        st.caption("Start the browser camera to run continuous, frame-by-frame AI detection.")
        webrtc_ctx = webrtc_streamer(
            key="fridgeguard-live-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: InspectionVideoProcessor(
                conf_threshold, iou_threshold
            ),
            # Required for browser-camera connections to a cloud-hosted app.
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_diag:
        st.markdown("#### Assembly Diagnostics")
        if webrtc_ctx.state.playing:
            st.success("Live camera is active. Bounding boxes and labels update on the video feed.")
            st.info("For the full component report and downloadable record, use Image Inspection.")
        else:
            status_card()

# ============================================================
# IMAGE INSPECTION + CLICKABLE CLASS FILTERING
# ============================================================
with tab_static:
    st.markdown("### Image Inspection")
    st.caption("Upload an image, then click one or more components in the results table to view only those detected classes.")

    uploaded_file = st.file_uploader(
        "Upload inspection image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        with st.spinner("Running AI assembly verification..."):
            result = analyze_image(image_array, conf_threshold, iou_threshold)

        col_img, col_report = st.columns([1.65, 1], gap="large")

        with col_img:
            st.markdown("#### AI Detection Result")
            image_placeholder = st.empty()

        with col_report:
            st.markdown("#### Assembly Decision")
            status_card(result)
            st.markdown("<br>", unsafe_allow_html=True)

            a, b = st.columns(2)
            with a:
                kpi_card("Verified", f'{result["total_verified"]}/{result["total_expected"]}')
            with b:
                kpi_card("Confidence", f'{result["avg_confidence"] * 100:.1f}%')

            st.markdown("#### Component Verification")
            report_df = build_report_dataframe(result)

            # RESTORED FEATURE: selecting table rows filters the displayed classes.
            selection_event = st.dataframe(
                report_df.drop(columns=["_class_name"]),
                use_container_width=True,
                hide_index=True,
                height=280,
                on_select="rerun",
                selection_mode="multi-row",
                column_config={
                    "Component": st.column_config.TextColumn("Component", width="medium"),
                    "Detected": st.column_config.NumberColumn("Detected", width="small"),
                    "Expected": st.column_config.NumberColumn("Expected", width="small"),
                    "Avg Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="medium"),
                }
            )

            if st.button("💾 Save Inspection Record", use_container_width=True):
                save_to_history(result)
                st.success("Inspection record saved successfully.")

        # Show filtered image when user clicks classes in table.
        selected_rows = selection_event.selection.rows

        if selected_rows:
            selected_classes = report_df.iloc[selected_rows]["_class_name"].tolist()
            filtered_img = filtered_class_image(
                image_array,
                selected_classes,
                conf_threshold,
                iou_threshold
            )
            image_placeholder.image(
                filtered_img,
                caption="Filtered Output — Selected Components Only",
                use_container_width=True
            )
        else:
            image_placeholder.image(
                result["annotated"],
                caption="Complete AI Detection Output",
                use_container_width=True
            )

        st.markdown("---")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            kpi_card("Expected Parts", result["total_expected"])
        with s2:
            kpi_card("Detected Parts", result["total_detected"])
        with s3:
            kpi_card("Average Confidence", f'{result["avg_confidence"] * 100:.1f}%')
        with s4:
            kpi_card("Inference Time", f'{result["inference_ms"]:.0f} ms')

# ============================================================
# PRODUCTION ANALYTICS
# ============================================================
with tab_analytics:
    st.markdown("### Production Analytics")
    df = st.session_state.history.copy()

    if df.empty:
        st.info("No inspection records yet. Save image inspection results to build production analytics.")
    else:
        total = len(df)
        passed = int((df["Status"] == "PASS").sum())
        failed = int((df["Status"] == "FAIL").sum())
        yield_rate = (passed / total * 100) if total else 0

        a, b, c, d = st.columns(4)
        with a:
            kpi_card("Total Inspections", total)
        with b:
            kpi_card("Units Passed", passed)
        with c:
            kpi_card("Units Failed", failed)
        with d:
            kpi_card("First Pass Yield", f"{yield_rate:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        chart_col, log_col = st.columns([1, 2], gap="large")

        with chart_col:
            st.markdown("#### Inspection Outcomes")
            status_counts = df["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
            st.bar_chart(status_counts, x="Status", y="Count", use_container_width=True)

        with log_col:
            st.markdown("#### Inspection History")
            st.dataframe(df, hide_index=True, use_container_width=True, height=320)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV Report",
                data=csv,
                file_name=f"FridgeGuard_AI_QA_Report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
