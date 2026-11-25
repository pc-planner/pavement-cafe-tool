import io
import re
import cv2
import easyocr
import fitz  # PyMuPDF for PDF uploads
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="Pavement Cafe Plan Generator", layout="wide")
st.title("Pavement Cafe Dimensioned Polygon Generator")


@st.cache_resource
def load_ocr():
  return easyocr.Reader(["en"], gpu=False)


reader = load_ocr()


def draw_dimension_arrow(draw, p1, p2, text, offset=25, color="black"):
  """Draws a clean architectural dimension line with extension ticks and label."""
  x1, y1 = p1
  x2, y2 = p2
  if y1 == y2:  # Horizontal dimension
    yd = y1 + offset
    draw.line([(x1, y1), (x1, yd)], fill="#64748b", width=1)
    draw.line([(x2, y2), (x2, yd)], fill="#64748b", width=1)
    draw.line([(x1, yd), (x2, yd)], fill=color, width=2)
    draw.line([(x1 - 4, yd + 4), (x1 + 4, yd - 4)], fill=color, width=2)
    draw.line([(x2 - 4, yd + 4), (x2 + 4, yd - 4)], fill=color, width=2)
    draw.text(((x1 + x2) // 2 - 18, yd + 5), text, fill=color)
  else:  # Vertical dimension
    xd = x1 + offset
    draw.line([(x1, y1), (xd, y1)], fill="#64748b", width=1)
    draw.line([(x2, y2), (xd, y2)], fill="#64748b", width=1)
    draw.line([(xd, y1), (xd, y2)], fill=color, width=2)
    draw.line([(xd - 4, y1 + 4), (xd + 4, y1 - 4)], fill=color, width=2)
    draw.line([(xd - 4, y2 + 4), (xd + 4, y2 - 4)], fill=color, width=2)
    draw.text((xd + 8, (y1 + y2) // 2 - 6), text, fill=color)


uploaded_file = st.file_uploader(
    "Upload Site Plan (JPG, PNG, WEBP, or PDF)",
    type=["jpg", "jpeg", "png", "webp", "pdf"],
)

if uploaded_file:
  file_bytes = uploaded_file.read()

  # Convert PDF page 1 or standard image into OpenCV format
  if uploaded_file.name.lower().endswith(".pdf"):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pix = doc.load_page(0).get_pixmap(dpi=150)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    img_cv = cv2.cvtColor(
        img_array, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR
    )
  else:
    np_arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

  # Resize large images so free Streamlit Cloud (1GB RAM) never runs out of memory
  h, w = img_cv.shape[:2]
  if w > 1200:
    img_cv = cv2.resize(img_cv, (1200, int(h * (1200 / w))))

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Uploaded Plan")
    st.image(img_cv, channels="BGR", use_container_width=True)

  # --- AUTOMATIC EXTRACTION (OpenCV + EasyOCR) ---
  with st.spinner("Reading dimensions and detecting boundary polygons..."):
    ocr_results = reader.readtext(img_cv)
    dimensions = []
    for bbox, text, _ in ocr_results:
      match = re.search(r"(\d+\.?\d*)", text)
      if match:
        val = float(match.group(1))
        if val > 100:  # Convert mm (e.g. 4500) to meters (4.5)
          val /= 1000.0
        if 0.2 <= val <= 50.0:  # Filter out unlikely numbers
          cx = int((bbox[0][0] + bbox[2][0]) / 2)
          cy = int((bbox[0][1] + bbox[2][1]) / 2)
          dimensions.append({"val": val, "x": cx, "y": cy})

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
      bx, by, bw, bh = cv2.boundingRect(c)
      if bw > 80 and bh > 40:
        boxes.append((bx, by, bw, bh))
    boxes.sort(key=lambda b: b[1])  # Sort top-to-bottom

  # Default fallback values
  adj_w, adj_d, has_remote, rem_w, rem_d, gap = 4.0, 1.5, False, 4.0, 1.5, 2.0

  if boxes and dimensions:
    bx, by, bw, bh = boxes[-1]  # Bottom box assumed to be Adjacent PC
    adj_w = min(
        dimensions,
        key=lambda d: (d["x"] - (bx + bw / 2)) ** 2 + (d["y"] - by) ** 2,
    )["val"]
    adj_d = min(
        dimensions,
        key=lambda d: (d["x"] - bx) ** 2 + (d["y"] - (by + bh / 2)) ** 2,
    )["val"]

    if len(boxes) > 1:
      has_remote = True
      rx, ry, rw, rh = boxes[0]  # Top box assumed to be Remote PC
      rem_w = min(
          dimensions,
          key=lambda d: (d["x"] - (rx + rw / 2)) ** 2 + (d["y"] - ry) ** 2,
      )["val"]
      rem_d = min(
          dimensions,
          key=lambda d: (d["x"] - rx) ** 2 + (d["y"] - (ry + rh / 2)) ** 2,
      )["val"]
      gap_y = (ry + rh + by) / 2
      gap = min(
          dimensions,
          key=lambda d: (d["x"] - (rx + rw / 2)) ** 2 + (d["y"] - gap_y) ** 2,
      )["val"]

  # --- AUTO-FILLED CONTROLS & RENDERER ---
  with col2:
    st.subheader("Dimensioned Polygon Output")
    c1, c2 = st.columns(2)
    adj_w = c1.number_input(
        "Adjacent PC Width (m)", value=float(adj_w), min_value=0.1, step=0.1
    )
    adj_d = c2.number_input(
        "Adjacent PC Depth (m)", value=float(adj_d), min_value=0.1, step=0.1
    )

    has_remote = st.checkbox("Include Remote PC", value=has_remote)
    if has_remote:
      c3, c4, c5 = st.columns(3)
      rem_w = c3.number_input(
          "Remote Width (m)", value=float(rem_w), min_value=0.1, step=0.1
      )
      rem_d = c4.number_input(
          "Remote Depth (m)", value=float(rem_d), min_value=0.1, step=0.1
      )
      gap = c5.number_input(
          "Separation Gap (m)", value=float(gap), min_value=0.1, step=0.1
      )

    # Auto-calculate scale so the drawing always fits neatly on an 800x800 canvas
    total_w_m = max(adj_w, rem_w) if has_remote else adj_w
    total_h_m = (adj_d + gap + rem_d) if has_remote else adj_d
    scale = min(520 / total_w_m, 520 / total_h_m)

    out = Image.new("RGB", (800, 800), "white")
    draw = ImageDraw.Draw(out)

    # Premises / Building Frontage reference line along the bottom
    ax0 = 120
    ay1 = 680
    ay0 = ay1 - int(adj_d * scale)
    ax1 = ax0 + int(adj_w * scale)

    draw.line([(60, ay1), (740, ay1)], fill="black", width=4)
    draw.text((65, ay1 + 12), "PREMISES / BUILDING FRONTAGE", fill="black")

    # Draw Adjacent PC Polygon
    draw.rectangle([ax0, ay0, ax1, ay1], outline="black", fill="#dbeafe", width=3)
    adj_area = round(adj_w * adj_d, 2)
    draw.text(
        (ax0 + 10, (ay0 + ay1) // 2 - 8),
        f"Adjacent PC ({adj_area} m²)",
        fill="black",
    )

    # External dimension lines for Adjacent PC
    draw_dimension_arrow(draw, (ax0, ay1), (ax1, ay1), f"{adj_w}m", offset=35)
    draw_dimension_arrow(draw, (ax1, ay0), (ax1, ay1), f"{adj_d}m", offset=25)

    # Draw Remote PC Polygon & Gap
    if has_remote:
      ry1 = ay0 - int(gap * scale)
      ry0 = ry1 - int(rem_d * scale)
      rx1 = ax0 + int(rem_w * scale)

      draw.rectangle(
          [ax0, ry0, rx1, ry1], outline="black", fill="#fef3c7", width=3
      )
      rem_area = round(rem_w * rem_d, 2)
      draw.text(
          (ax0 + 10, (ry0 + ry1) // 2 - 8),
          f"Remote PC ({rem_area} m²)",
          fill="black",
      )

      # External dimension lines for Remote PC
      draw_dimension_arrow(draw, (ax0, ry0), (rx1, ry0), f"{rem_w}m", offset=-25)
      draw_dimension_arrow(draw, (rx1, ry0), (rx1, ry1), f"{rem_d}m", offset=25)

      # Separation gap arrow between Adjacent and Remote zones
      gap_x = ax0 + int(min(adj_w, rem_w) * scale * 0.5)
      draw.line([(gap_x, ry1), (gap_x, ay0)], fill="#dc2626", width=2)
      draw.text(
          (gap_x + 8, (ry1 + ay0) // 2 - 6),
          f"Clear Gap: {gap}m",
          fill="#dc2626",
      )

      total_area = round(adj_area + rem_area, 2)
      draw.text(
          (60, 30), f"Total Pavement Cafe Area: {total_area} m²", fill="black"
      )
    else:
      draw.text(
          (60, 30), f"Total Pavement Cafe Area: {adj_area} m²", fill="black"
      )

    st.image(out, use_container_width=True)

    # Export to JPEG
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=95)
    st.download_button(
        label="Download Dimensioned JPEG",
        data=buf.getvalue(),
        file_name="pavement_cafe_plan.jpg",
        mime="image/jpeg",
    )
