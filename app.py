"""
app.py — Enhanced Live Surveillance System Dashboard.

Streamlit-based UI with:
  • Config sidebar (confidence, cooldown, camera source, alerts)
  • Live feed with YOLOv8 detection + annotated zones
  • ROI zone configuration via drawable canvas
  • Analytics charts (Plotly)
  • Snapshot gallery
  • Detection log viewer
"""

import os
import time
import json

import cv2
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from detection_engine import DetectionEngine, Zone
from frame_manager import FrameManager
from alert_engine import AlertEngine
from db_logger import DBLogger
from snapshot_saver import SnapshotSaver

# ── Page config ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Surveillance System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ──────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0; font-size: 1.8rem; font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0; opacity: 0.8; font-size: 0.95rem;
    }

    .status-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        color: white;
        text-align: center;
    }
    .status-card .value {
        font-size: 1.8rem; font-weight: 700; color: #00d4ff;
    }
    .status-card .label {
        font-size: 0.8rem; opacity: 0.7; text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1> Enhanced Live Surveillance System</h1>
    <p>Real-time object detection • Multi-zone monitoring • Intelligent alerts</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
#   SIDEBAR — Configuration
# ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("##  Configuration")

    st.markdown("###  Camera Source")
    source_type = st.selectbox(
        "Input type",
        ["Webcam", "RTSP / IP Camera", "Video File"],
        key="source_type",
    )

    if source_type == "Webcam":
        cam_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0)
        cam_source = int(cam_index)
    elif source_type == "RTSP / IP Camera":
        cam_source = st.text_input(
            "RTSP URL",
            placeholder="rtsp://user:pass@192.168.1.100:554/stream",
        )
    else:
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        cam_source = None
        if uploaded:
            tmp_path = os.path.join("temp_video.mp4")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            cam_source = tmp_path

    st.markdown("---")
    st.markdown("###  Detection Settings")
    confidence_threshold = st.slider(
        "Confidence threshold", 0.10, 1.0, 0.50, 0.05, key="conf"
    )
    target_classes = st.multiselect(
        "Classes to detect",
        ["person", "car", "bicycle", "motorcycle", "bus", "truck",
         "cat", "dog", "backpack", "handbag", "suitcase"],
        default=["person"],
        key="classes",
    )
    frame_skip = st.slider("Process every N frames", 1, 10, 2, key="skip")

    st.markdown("---")
    st.markdown("###  Alert Settings")
    cooldown = st.slider(
        "Alert cooldown (seconds)", 5, 300, 30, 5, key="cooldown"
    )
    alarm_path = st.text_input(
        "Alarm sound file path",
        value="",
        placeholder="path/to/alarm.mp3",
        key="alarm",
    )
    email_enabled = st.checkbox("Enable email alerts", key="email_en")
    sms_enabled = st.checkbox("Enable SMS alerts (Twilio)", key="sms_en")

    if email_enabled:
        email_to = st.text_input("Alert email recipient", key="email_to")
    else:
        email_to = ""

    if sms_enabled:
        sms_to = st.text_input("Alert SMS recipient", key="sms_to")
    else:
        sms_to = ""

# ────────────────────────────────────────────────────────────────
#   INITIALIZE ENGINES (cached with st.session_state)
# ────────────────────────────────────────────────────────────────

if "engine" not in st.session_state:
    st.session_state.engine = DetectionEngine()

if "db" not in st.session_state:
    st.session_state.db = DBLogger()

if "snapper" not in st.session_state:
    st.session_state.snapper = SnapshotSaver()

if "zones" not in st.session_state:
    st.session_state.zones = []

if "surveillance_active" not in st.session_state:
    st.session_state.surveillance_active = False

engine: DetectionEngine = st.session_state.engine
db: DBLogger = st.session_state.db
snapper: SnapshotSaver = st.session_state.snapper

# Build alert engine fresh each run (picks up sidebar changes)
alert = AlertEngine(
    cooldown_seconds=cooldown,
    alarm_sound_path=alarm_path if alarm_path else None,
    email_enabled=email_enabled,
    sms_enabled=sms_enabled,
)

# ────────────────────────────────────────────────────────────────
#   TABS
# ────────────────────────────────────────────────────────────────

tab_live, tab_roi, tab_analytics, tab_snapshots, tab_logs = st.tabs(
    [" Live Feed", " ROI Zones", " Analytics", " Snapshots", " Logs"]
)

# ════════════════════════════════════════════════════════════════
#   TAB 1 — Live Feed
# ════════════════════════════════════════════════════════════════

with tab_live:
    col_ctrl, col_status = st.columns([1, 3])

    with col_ctrl:
        start_btn = st.button("▶ Start Surveillance", key="start_btn", type="primary",
                              use_container_width=True)
        stop_btn = st.button("⏹ Stop Surveillance", key="stop_btn",
                             use_container_width=True)

    with col_status:
        status_cols = st.columns(4)
        status_placeholder_fps = status_cols[0].empty()
        status_placeholder_det = status_cols[1].empty()
        status_placeholder_alerts = status_cols[2].empty()
        status_placeholder_total = status_cols[3].empty()

    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    # Build zone list from session state
    zones = [
        Zone(name=z["name"], x1=z["x1"], y1=z["y1"], x2=z["x2"], y2=z["y2"],
             color=tuple(z.get("color", [255, 0, 0])))
        for z in st.session_state.zones
    ]

    if start_btn and cam_source is not None:
        st.session_state.surveillance_active = True

    if stop_btn:
        st.session_state.surveillance_active = False

    if st.session_state.surveillance_active and cam_source is not None:
        fm = FrameManager(
            source=cam_source,
            process_every_n=frame_skip,
            resolution=(640, 480),
        )
        if not fm.start():
            st.error("❌ Unable to open camera source. Check your settings.")
            st.session_state.surveillance_active = False
        else:
            info_placeholder.info("🟢 Surveillance is running… switch tabs freely.")
            detection_count = 0
            alert_count = 0

            try:
                while st.session_state.surveillance_active:
                    frame = fm.get_frame(timeout=2.0)
                    if frame is None:
                        if not fm.is_running:
                            info_placeholder.warning("📹 Video stream ended.")
                            break
                        continue

                    # Run detection
                    detections = engine.detect(
                        frame,
                        confidence_threshold=confidence_threshold,
                        target_classes=target_classes if target_classes else None,
                        zones=zones,
                    )

                    # Annotate
                    annotated = engine.annotate_frame(frame, detections, zones)

                    # Process zone alerts
                    zone_detections = [d for d in detections if d.in_zone]
                    if zone_detections:
                        detection_count += len(zone_detections)
                        # Save snapshot
                        snap_path = snapper.save(annotated)

                        for d in zone_detections:
                            # Log to DB
                            db.log_detection(
                                zone_name=d.in_zone,
                                detected_class=d.class_name,
                                confidence=d.confidence,
                                snapshot_path=snap_path,
                            )
                            # Trigger alert (respects cooldown)
                            fired = alert.trigger(
                                zone_name=d.in_zone,
                                class_name=d.class_name,
                                confidence=d.confidence,
                                email_to_override=email_to,
                                sms_to_override=sms_to,
                            )
                            if fired:
                                alert_count += 1

                    # Display frame
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_container_width=True)

                    # Update status cards
                    status_placeholder_fps.markdown(
                        f'<div class="status-card"><div class="value">'
                        f'{fm.fps:.1f}</div><div class="label">FPS</div></div>',
                        unsafe_allow_html=True,
                    )
                    status_placeholder_det.markdown(
                        f'<div class="status-card"><div class="value">'
                        f'{detection_count}</div><div class="label">Zone Detections</div></div>',
                        unsafe_allow_html=True,
                    )
                    status_placeholder_alerts.markdown(
                        f'<div class="status-card"><div class="value">'
                        f'{alert_count}</div><div class="label">Alerts Fired</div></div>',
                        unsafe_allow_html=True,
                    )
                    status_placeholder_total.markdown(
                        f'<div class="status-card"><div class="value">'
                        f'{db.get_total_count()}</div><div class="label">Total Logged</div></div>',
                        unsafe_allow_html=True,
                    )

                    time.sleep(0.01)  # yield to Streamlit event loop

            finally:
                fm.stop()
                st.session_state.surveillance_active = False
                info_placeholder.success("⏹ Surveillance stopped.")

# ════════════════════════════════════════════════════════════════
#   TAB 2 — ROI Zone Configuration
# ════════════════════════════════════════════════════════════════

with tab_roi:
    st.markdown("###  Configure Monitoring Zones")
    st.markdown(
        "Define rectangular monitoring zones using the controls below. "
        "Each zone triggers independent alerts when objects are detected inside it."
    )

    # ── Number of zones ──
    if "num_zones" not in st.session_state:
        st.session_state.num_zones = len(st.session_state.zones) if st.session_state.zones else 1

    col_add, col_remove = st.columns(2)
    with col_add:
        if st.button("➕ Add Zone", use_container_width=True):
            st.session_state.num_zones += 1
            st.rerun()
    with col_remove:
        if st.button("➖ Remove Last Zone", use_container_width=True) and st.session_state.num_zones > 0:
            st.session_state.num_zones = max(0, st.session_state.num_zones - 1)
            st.rerun()

    zone_colors_hex = ["#FF4444", "#44FF44", "#FFA500", "#FF44FF", "#FFFF00", "#44FFFF"]
    zone_colors_bgr = [
        [0, 0, 255], [0, 255, 0], [0, 165, 255],
        [255, 0, 255], [0, 255, 255], [255, 255, 0],
    ]

    new_zones = []

    for i in range(st.session_state.num_zones):
        color_hex = zone_colors_hex[i % len(zone_colors_hex)]
        with st.expander(f"🟥 Zone {i + 1}", expanded=True):
            # Load defaults from saved zones if available
            saved = st.session_state.zones[i] if i < len(st.session_state.zones) else None

            zone_name = st.text_input(
                "Zone name",
                value=saved["name"] if saved else f"Zone {i + 1}",
                key=f"zname_{i}",
            )

            zc1, zc2 = st.columns(2)
            with zc1:
                zx1 = st.number_input("X1 (left)", 0, 640, value=saved["x1"] if saved else 100, step=10, key=f"zx1_{i}")
                zy1 = st.number_input("Y1 (top)", 0, 480, value=saved["y1"] if saved else 80, step=10, key=f"zy1_{i}")
            with zc2:
                zx2 = st.number_input("X2 (right)", 0, 640, value=saved["x2"] if saved else 400, step=10, key=f"zx2_{i}")
                zy2 = st.number_input("Y2 (bottom)", 0, 480, value=saved["y2"] if saved else 350, step=10, key=f"zy2_{i}")

            if zx2 > zx1 and zy2 > zy1:
                new_zones.append({
                    "name": zone_name,
                    "x1": zx1, "y1": zy1,
                    "x2": zx2, "y2": zy2,
                    "color": zone_colors_bgr[i % len(zone_colors_bgr)],
                })
            else:
                st.warning("⚠️ X2 must be > X1 and Y2 must be > Y1")

    # ── Visual preview ──
    st.markdown("####  Zone Preview (640 × 480)")
    preview = np.full((480, 640, 3), (30, 30, 50), dtype=np.uint8)
    for z in new_zones:
        clr = tuple(z["color"])
        overlay = preview.copy()
        cv2.rectangle(overlay, (z["x1"], z["y1"]), (z["x2"], z["y2"]), clr, -1)
        cv2.addWeighted(overlay, 0.25, preview, 0.75, 0, preview)
        cv2.rectangle(preview, (z["x1"], z["y1"]), (z["x2"], z["y2"]), clr, 2)
        cv2.putText(preview, z["name"], (z["x1"] + 5, z["y1"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)

    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    st.image(preview_rgb, use_container_width=True)

    # ── Save button ──
    if st.button(" Save Zones", type="primary", use_container_width=True, key="save_zones"):
        st.session_state.zones = new_zones
        st.success(f"✅ Saved {len(new_zones)} zone(s)!")

    # ── Show saved zones ──
    if st.session_state.zones:
        st.markdown("#### ✅ Saved Zones")
        for z in st.session_state.zones:
            st.markdown(
                f"**{z['name']}** — "
                f"({z['x1']},{z['y1']}) → ({z['x2']},{z['y2']})"
            )

# ════════════════════════════════════════════════════════════════
#   TAB 3 — Analytics
# ════════════════════════════════════════════════════════════════

with tab_analytics:
    st.markdown("###  Detection Analytics")

    if st.button(" Refresh Analytics", key="refresh_analytics"):
        pass  # forces rerun

    analytics_cols = st.columns(3)

    # ── Metric cards ──
    total = db.get_total_count()
    zone_counts = db.get_zone_counts()
    class_counts = db.get_class_counts()

    analytics_cols[0].metric("Total Detections", total)
    analytics_cols[1].metric("Zones with Activity", len(zone_counts))
    analytics_cols[2].metric("Unique Classes", len(class_counts))

    chart_col1, chart_col2 = st.columns(2)

    # ── Hourly detections bar chart ──
    with chart_col1:
        hourly = db.get_hourly_counts(hours=24)
        if hourly:
            hours, counts = zip(*hourly)
            fig = px.bar(
                x=hours, y=counts,
                labels={"x": "Hour", "y": "Detections"},
                title="Detections per Hour (Last 24h)",
                color_discrete_sequence=["#00d4ff"],
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly data yet. Start surveillance to generate data.")

    # ── Zone distribution pie chart ──
    with chart_col2:
        if zone_counts:
            zone_names, zone_vals = zip(*zone_counts)
            fig = px.pie(
                names=zone_names, values=zone_vals,
                title="Detections by Zone",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No zone data yet.")

    # ── Class distribution bar chart ──
    if class_counts:
        cls_names, cls_vals = zip(*class_counts)
        fig = px.bar(
            x=cls_names, y=cls_vals,
            labels={"x": "Class", "y": "Count"},
            title="Detections by Class",
            color_discrete_sequence=["#ff6b6b"],
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
#   TAB 4 — Snapshot Gallery
# ════════════════════════════════════════════════════════════════

with tab_snapshots:
    st.markdown("###  Detection Snapshots")

    if st.button(" Refresh Gallery", key="refresh_gallery"):
        pass

    snapshots = snapper.get_recent(n=30)

    if snapshots:
        cols = st.columns(4)
        for i, snap_path in enumerate(snapshots):
            with cols[i % 4]:
                try:
                    img = Image.open(snap_path)
                    st.image(img, use_container_width=True)
                    basename = os.path.basename(snap_path)
                    # Extract timestamp from filename: snapshot_YYYYMMDD_HHMMSS_ffffff.jpg
                    ts_str = basename.replace("snapshot_", "").replace(".jpg", "")
                    parts = ts_str.split("_")
                    if len(parts) >= 2:
                        date_str = parts[0]
                        time_str = parts[1]
                        display = (
                            f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} "
                            f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                        )
                    else:
                        display = basename
                    st.caption(display)
                except Exception:
                    st.caption(f"⚠️ Could not load {os.path.basename(snap_path)}")
    else:
        st.info("No snapshots yet. Snapshots are saved automatically when objects are detected in a monitored zone.")

# ════════════════════════════════════════════════════════════════
#   TAB 5 — Detection Logs
# ════════════════════════════════════════════════════════════════

with tab_logs:
    st.markdown("###  Detection Log")

    log_cols = st.columns([1, 1, 2])
    with log_cols[0]:
        if st.button(" Refresh Logs", key="refresh_logs"):
            pass
    with log_cols[1]:
        log_limit = st.selectbox("Show last", [25, 50, 100, 200], index=1, key="log_limit")

    records = db.get_recent(limit=log_limit)

    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        # Reorder columns for display
        display_cols = ["id", "timestamp", "zone_name", "detected_class", "confidence"]
        available = [c for c in display_cols if c in df.columns]
        df = df[available]
        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].apply(lambda x: f"{x:.1%}")

        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No detection logs yet. Logs appear here when objects are detected in monitored zones.")

    # ── Clear logs ──
    with st.expander("⚠️ Danger Zone"):
        if st.button(" Clear All Logs", type="secondary"):
            db.clear_all()
            st.success("All logs cleared.")
            st.rerun()
