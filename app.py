from __future__ import annotations

import base64
from time import perf_counter

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.convolution import process_image


st.set_page_config(page_title="Clear-Pixel", page_icon=":camera:", layout="wide")


def inject_page_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --page-bg: linear-gradient(180deg, #f3efe7 0%, #e8f0ec 45%, #f4f7fb 100%);
            --panel-bg: rgba(255, 255, 255, 0.96);
            --panel-border: rgba(148, 163, 184, 0.34);
            --text-main: #172033;
            --text-muted: #4f5d75;
            --accent: #0f766e;
            --accent-soft: #dff6f2;
            --accent-strong: #b45309;
            --shadow: 0 20px 42px rgba(15, 23, 42, 0.12);
          }

          [data-testid="stAppViewContainer"] {
            background: var(--page-bg);
          }

          [data-testid="stHeader"] {
            background: transparent;
          }

          [data-testid="stSidebar"] {
            background:
              radial-gradient(circle at top, rgba(15, 118, 110, 0.12), transparent 36%),
              linear-gradient(180deg, #f9f7f1 0%, #eef5f1 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.28);
            box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.55);
          }

          [data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
          }

          .hero-panel,
          .empty-state,
          .section-panel {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            box-shadow: var(--shadow);
          }

          .hero-panel *,
          .empty-state *,
          .sidebar-wrap * {
            position: relative;
            z-index: 1;
          }

          .hero-panel {
            padding: 1.8rem 1.9rem;
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
          }

          .hero-panel::after {
            content: "";
            position: absolute;
            inset: auto -60px -90px auto;
            width: 200px;
            height: 200px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.18), transparent 68%);
          }

          .hero-kicker,
          .panel-kicker {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--accent);
            background: var(--accent-soft);
          }

          .hero-title {
            margin: 0.95rem 0 0.45rem;
            font-size: 2.65rem;
            line-height: 1.05;
            font-weight: 800;
            color: var(--text-main);
          }

          .hero-copy,
          .panel-copy,
          .sidebar-copy,
          .empty-copy {
            color: var(--text-muted);
            font-size: 1rem;
            line-height: 1.65;
          }

          .hero-grid,
          .empty-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.8rem;
            margin-top: 1.1rem;
          }

          .hero-chip,
          .empty-card {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: #fcfdff;
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
          }

          .hero-chip strong,
          .empty-card strong,
          .sidebar-card strong {
            display: block;
            margin-bottom: 0.2rem;
            color: var(--text-main);
          }

          .empty-state {
            padding: 2rem;
            margin-top: 1rem;
          }

          .empty-title {
            margin: 0.9rem 0 0.4rem;
            font-size: 2rem;
            line-height: 1.1;
            color: var(--text-main);
          }

          .empty-note {
            margin-top: 1rem;
            padding: 0.95rem 1rem;
            border-left: 4px solid var(--accent-strong);
            background: rgba(180, 83, 9, 0.08);
            border-radius: 14px;
            color: #7c4a13;
            font-size: 0.95rem;
          }

          .sidebar-wrap {
            padding: 0.2rem 0 0.6rem;
          }

          .sidebar-title {
            margin: 0.9rem 0 0.35rem;
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text-main);
          }

          .sidebar-card {
            margin: 1rem 0 1.25rem;
            padding: 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
          }

          .sidebar-meta {
            margin-top: 0.7rem;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            color: var(--text-muted);
            font-size: 0.93rem;
          }

          .stButton > button,
          .stDownloadButton > button {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%);
            color: white;
            font-weight: 700;
            padding: 0.6rem 1.25rem;
            box-shadow: 0 12px 26px rgba(29, 78, 216, 0.18);
          }

          .stFileUploader,
          .stSelectbox,
          .stSlider {
            background: rgba(255, 255, 255, 0.98);
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            padding: 0.35rem 0.45rem 0.55rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
          }

          [data-testid="stFileUploaderDropzone"] {
            background: #f8fafc;
            border: 1px dashed rgba(71, 85, 105, 0.45);
          }

          [data-testid="stBaseButton-secondary"] {
            background: #ffffff;
            color: var(--text-main);
            border: 1px solid rgba(148, 163, 184, 0.35);
          }

          [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(148, 163, 184, 0.22);
            padding: 1rem;
            border-radius: 18px;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
          }

          [data-testid="stMetricLabel"],
          [data-testid="stMetricLabel"] *,
          [data-testid="stMetricValue"],
          [data-testid="stMetricValue"] * {
            color: var(--text-main) !important;
          }

          [data-testid="stMetricLabel"] {
            opacity: 0.88;
          }

          [data-testid="stWidgetLabel"] p,
          .stCaption,
          .stMarkdown p,
          label,
          .sidebar-copy,
          .empty-copy {
            color: var(--text-muted) !important;
            opacity: 1 !important;
          }

          .hero-title,
          .hero-copy,
          .hero-chip,
          .hero-chip strong,
          .empty-title,
          .empty-card,
          .empty-card strong,
          .sidebar-title,
          .sidebar-card,
          .sidebar-card strong {
            opacity: 1 !important;
            color: var(--text-main) !important;
          }

          .stInfo {
            background: linear-gradient(180deg, #d6ecff 0%, #c8e4fb 100%);
            border: 1px solid rgba(96, 165, 250, 0.3);
            color: #12314d;
          }

          .stInfo p {
            color: #174166 !important;
          }

          .stSelectbox [data-baseweb="select"] > div,
          .stFileUploader section,
          .stSlider > div[data-baseweb="slider"] {
            background: transparent;
          }

          [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: var(--text-main);
          }

          [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.14);
          }

          @media (max-width: 900px) {
            .hero-title {
              font-size: 2.1rem;
            }

            .empty-title {
              font-size: 1.7rem;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_download_bytes(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode the processed image.")
    return buffer.tobytes()


def resize_for_preview(
    image: np.ndarray,
    max_width: int = 760,
    max_height: int = 460,
) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image
    scale = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def encode_image_for_html(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode image for comparison view.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def render_comparison_slider(
    original_image: np.ndarray,
    processed_image: np.ndarray,
    *,
    max_preview_width: int = 760,
    max_preview_height: int = 460,
) -> None:
    preview_original = resize_for_preview(original_image, max_preview_width, max_preview_height)
    preview_processed = resize_for_preview(processed_image, max_preview_width, max_preview_height)
    preview_height, preview_width = preview_original.shape[:2]
    component_height = preview_height + 54

    original_encoded = encode_image_for_html(preview_original)
    processed_encoded = encode_image_for_html(preview_processed)
    slider_id = f"compare-{preview_width}-{preview_height}-{abs(int(preview_processed.sum()) - int(preview_original.sum()))}"

    html_markup = f"""
    <div class="comparison-shell">
      <div class="comparison-frame" id="{slider_id}">
        <svg
          viewBox="0 0 {preview_width} {preview_height}"
          class="comparison-canvas"
          aria-label="Before and after image comparison"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <clipPath id="{slider_id}-clip">
              <rect id="{slider_id}-clip-rect" x="0" y="0" width="{preview_width / 2}" height="{preview_height}" />
            </clipPath>
          </defs>
          <image href="data:image/png;base64,{original_encoded}" x="0" y="0" width="{preview_width}" height="{preview_height}" />
          <image
            href="data:image/png;base64,{processed_encoded}"
            x="0"
            y="0"
            width="{preview_width}"
            height="{preview_height}"
            clip-path="url(#{slider_id}-clip)"
          />
          <line
            id="{slider_id}-divider"
            x1="{preview_width / 2}"
            y1="0"
            x2="{preview_width / 2}"
            y2="{preview_height}"
            stroke="rgba(255,255,255,0.95)"
            stroke-width="4"
          />
          <g id="{slider_id}-handle" transform="translate({preview_width / 2}, {preview_height / 2})">
            <circle r="28" fill="#ffffff" />
            <text
              text-anchor="middle"
              dominant-baseline="middle"
              font-size="19"
              font-family="Arial, sans-serif"
              font-weight="700"
              fill="#111827"
            >&lt;&gt;</text>
          </g>
        </svg>
      </div>
      <div class="comparison-labels">
        <span>Original</span>
        <span>Processed</span>
      </div>
    </div>

    <style>
      .comparison-shell {{
        max-width: {preview_width}px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
      }}
      .comparison-frame {{
        position: relative;
        width: 100%;
        border-radius: 18px;
        overflow: hidden;
        background: #111827;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.18);
        cursor: ew-resize;
        touch-action: none;
      }}
      .comparison-canvas {{
        display: block;
        width: 100%;
        height: auto;
        user-select: none;
      }}
      .comparison-labels {{
        display: flex;
        justify-content: space-between;
        color: #475569;
        font-size: 0.9rem;
        margin-top: 0.55rem;
        padding: 0 0.25rem;
      }}
    </style>

    <script>
      const frame = document.getElementById("{slider_id}");
      const clipRect = document.getElementById("{slider_id}-clip-rect");
      const divider = document.getElementById("{slider_id}-divider");
      const handle = document.getElementById("{slider_id}-handle");
      const svgWidth = {preview_width};
      const svgHeight = {preview_height};
      let isDragging = false;

      const updateSlider = (clientX) => {{
        const rect = frame.getBoundingClientRect();
        const relativeX = Math.min(Math.max(clientX - rect.left, 0), rect.width);
        const svgX = (relativeX / rect.width) * svgWidth;
        clipRect.setAttribute("width", svgX);
        divider.setAttribute("x1", svgX);
        divider.setAttribute("x2", svgX);
        handle.setAttribute("transform", `translate(${{svgX}}, ${{svgHeight / 2}})`);
      }};

      const beginDrag = (event) => {{
        isDragging = true;
        updateSlider(event.clientX);
      }};

      const onDrag = (event) => {{
        if (!isDragging) {{
          return;
        }}
        updateSlider(event.clientX);
      }};

      const endDrag = () => {{
        isDragging = false;
      }};

      frame.addEventListener("pointerdown", (event) => {{
        beginDrag(event);
        frame.setPointerCapture(event.pointerId);
      }});
      frame.addEventListener("pointermove", onDrag);
      frame.addEventListener("pointerup", (event) => {{
        endDrag();
        frame.releasePointerCapture(event.pointerId);
      }});
      frame.addEventListener("pointerleave", endDrag);
    </script>
    """

    components.html(html_markup, height=component_height, scrolling=False)


def render_sidebar_intro() -> None:
    st.markdown(
        """
        <div class="sidebar-wrap">
          <span class="panel-kicker">Control deck</span>
          <div class="sidebar-title">Tune the effect</div>
          <p class="sidebar-copy">
            Upload a photo, choose the processing style, and adjust the kernel to control how soft or crisp the result becomes.
          </p>
          <div class="sidebar-card">
            <strong>Preset feel</strong>
            <div class="sidebar-meta">
              <span><strong>Blur</strong> Gentle smoothing for a softer image.</span>
              <span><strong>Sharpen</strong> Adds local contrast for a clearer look.</span>
              <span><strong>Larger kernels</strong> Push the effect further each step.</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
          <span class="panel-kicker">Ready when you are</span>
          <h2 class="empty-title">Drop in an image to preview the transformation.</h2>
          <p class="empty-copy">
            Clear-Pixel lets you compare the original and processed result side by side, then download the final output once it looks right.
          </p>
          <div class="empty-grid">
            <div class="empty-card">
              <strong>1. Upload</strong>
              Add a PNG or JPG from the sidebar to begin.
            </div>
            <div class="empty-card">
              <strong>2. Adjust</strong>
              Switch between blur and sharpen, then increase the kernel for a stronger effect.
            </div>
            <div class="empty-card">
              <strong>3. Compare</strong>
              Use the slider to inspect details before exporting the result.
            </div>
          </div>
          <div class="empty-note">
            Tip: start with a kernel size of 3 or 5 for subtle edits, then scale up only if the effect still feels too light.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_page_styles()

st.markdown(
    """
    <div class="hero-panel">
      <span class="hero-kicker">Interactive image lab</span>
      <h1 class="hero-title">Clear-Pixel</h1>
      <p class="hero-copy">
        Explore matrix convolution through a cleaner, more visual workflow for soft blur, edge-focused sharpening, and quick before-and-after review.
      </p>
      <div class="hero-grid">
        <div class="hero-chip"><strong>Live comparison</strong>Drag the center divider to inspect detail changes.</div>
        <div class="hero-chip"><strong>Kernel control</strong>Scale the effect with odd-sized convolution kernels.</div>
        <div class="hero-chip"><strong>Quick export</strong>Download the processed image as a PNG when it looks right.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

feature_columns = st.columns(3)
feature_columns[0].info("Upload an image and test convolution interactively.")
feature_columns[1].info("Switch between blur and sharpen with adjustable kernel size.")
feature_columns[2].info("Inspect the result, compare it, and download the output.")

with st.sidebar:
    render_sidebar_intro()
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, and JPEG.",
    )
    mode = st.selectbox(
        "Processing style",
        options=["blur", "sharpen"],
        format_func=lambda option: "Soft Blur" if option == "blur" else "Crisp Sharpen",
    )
    kernel_size = st.slider("Kernel intensity", min_value=3, max_value=15, step=2, value=3)
    st.caption("Lower values keep the effect subtle. Higher values make the blur or sharpening more dramatic.")

if uploaded_file is not None:
    try:
        original_image = decode_uploaded_image(uploaded_file.getvalue())
        start_time = perf_counter()
        processed_image, kernel = process_image(original_image, mode, kernel_size)
        processing_time_ms = (perf_counter() - start_time) * 1000
        preview_original = resize_for_preview(original_image, max_width=520, max_height=360)
        preview_processed = resize_for_preview(processed_image, max_width=520, max_height=360)

        stats_columns = st.columns(4)
        stats_columns[0].metric("Operation", mode.title())
        stats_columns[1].metric("Kernel Size", f"{kernel_size} x {kernel_size}")
        stats_columns[2].metric("Processing Time", f"{processing_time_ms:.2f} ms")
        stats_columns[3].metric("Image Size", f"{original_image.shape[1]} x {original_image.shape[0]}")

        st.markdown("#### Before and after comparison")
        st.caption("Drag the center handle to compare the original image with the processed result.")
        render_comparison_slider(original_image, processed_image)

        preview_columns = st.columns(2)

        with preview_columns[0]:
            st.markdown("#### Original image")
            st.image(to_rgb(preview_original), width=preview_original.shape[1])

        with preview_columns[1]:
            st.markdown("#### Processed image")
            st.image(to_rgb(preview_processed), width=preview_processed.shape[1])

        st.markdown("#### Kernel used")
        st.code(np.array2string(kernel, precision=3), language="text")

        st.download_button(
            label="Download output image",
            data=to_download_bytes(processed_image),
            file_name=f"{mode}_output.png",
            mime="image/png",
        )
    except ValueError as error:
        st.error(str(error))
else:
    render_empty_state()
