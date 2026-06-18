from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from PIL import ImageDraw

try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is not installed. Install it with: python -m pip install streamlit"
    ) from exc

sys.path.append(str(Path(__file__).resolve().parent))

from verifier import (  # noqa: E402
    CLASS_MAP,
    calculate_relative_location,
    class_to_zh,
    generate_feedback,
    overall_status,
    verify_diagnosis,
)
from voc_reader import (  # noqa: E402
    DATASET_DIR,
    crop_patch,
    dataset_exists,
    draw_bbox,
    list_image_ids,
    load_image,
    parse_annotation,
    read_split,
)


CASE_FILE = Path(__file__).resolve().parent / "diagnosis_cases.json"
GRID_LABELS = [
    "图像左上区域",
    "图像中上区域",
    "图像右上区域",
    "图像左中区域",
    "图像中心区域",
    "图像右中区域",
    "图像左下区域",
    "图像中下区域",
    "图像右下区域",
]


def load_cases() -> dict:
    return json.loads(CASE_FILE.read_text(encoding="utf-8"))


def opposite_location(location: str) -> str:
    mapping = {
        "图像左上区域": "图像右下区域",
        "图像中上区域": "图像中下区域",
        "图像右上区域": "图像左下区域",
        "图像左中区域": "图像右中区域",
        "图像中心区域": "图像左下区域",
        "图像右中区域": "图像左中区域",
        "图像左下区域": "图像右上区域",
        "图像中下区域": "图像中上区域",
        "图像右下区域": "图像左上区域",
    }
    return mapping.get(location, "图像左下区域")


def wrong_class(true_class: str) -> str:
    for key, value in CLASS_MAP.items():
        if key != true_class:
            return value
    return "雷击损伤"


def materialize_case(template: dict, true_class: str, true_location: str) -> dict:
    payload = json.loads(json.dumps(template["diagnosis"], ensure_ascii=False))
    replacements = {
        "__TRUE_CLASS__": class_to_zh(true_class),
        "__WRONG_CLASS__": wrong_class(true_class),
        "__TRUE_LOCATION__": true_location,
        "__WRONG_LOCATION__": opposite_location(true_location),
    }
    for key, value in list(payload.items()):
        if isinstance(value, str) and value in replacements:
            payload[key] = replacements[value]
    return payload


def draw_location_grid(image, bbox, location: str):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    for x in (w / 3, 2 * w / 3):
        draw.line((x, 0, x, h), fill="#38bdf8", width=3)
    for y in (h / 3, 2 * h / 3):
        draw.line((0, y, w, y), fill="#38bdf8", width=3)
    cx, cy = bbox.center
    r = 12
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#e11d48", outline="white", width=3)
    draw.rectangle(bbox.as_tuple(), outline="#e11d48", width=5)
    return canvas


def result_dataframe(results):
    return pd.DataFrame(
        [
            {
                "属性": r.attribute,
                "MLLM 描述": r.diagnosis,
                "参考证据": r.evidence,
                "结果": r.result,
            }
            for r in results
        ]
    )


def section_card(title: str, body: str, accent: str = "#2563eb") -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid #d9e2ec;
            border-left: 5px solid {accent};
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            background: #ffffff;
            min-height: 116px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        ">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.35rem;">{title}</div>
            <div style="font-size: 0.92rem; line-height: 1.55; color: #334155;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str, accent: str = "#0f766e") -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            min-height: 112px;
        ">
            <div style="font-size: 0.78rem; color: #64748b;">{label}</div>
            <div style="font-size: 1.55rem; font-weight: 750; color: {accent}; line-height: 1.2;">{value}</div>
            <div style="font-size: 0.86rem; color: #475569; margin-top: 0.35rem;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_block_flow(stage: str = "模型结构总览") -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid #d9e2ec;
            border-radius: 10px;
            padding: 1rem;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <div style="font-weight:800; color:#0f172a;">{stage}</div>
                <div style="font-size:0.82rem; color:#64748b;">AEGIS-Net</div>
            </div>
            <div style="display:grid; grid-template-columns: 1.1fr 0.22fr 1.1fr 0.22fr 1fr; gap:0.55rem; align-items:stretch;">
                <div style="border:1px solid #93c5fd; border-radius:8px; padding:0.8rem; background:linear-gradient(160deg,#eff6ff,#dbeafe); box-shadow: 8px 8px 0 rgba(37,99,235,0.12);">
                    <div style='font-weight:700;'>视觉输入</div>
                    <div style='font-size:0.82rem; color:#334155; margin-top:0.35rem;'>原图 + patch + bbox</div>
                </div>
                <div style="display:flex; align-items:center; justify-content:center; font-size:1.35rem; color:#64748b;">→</div>
                <div style="border:1px solid #c4b5fd; border-radius:8px; padding:0.8rem; background:linear-gradient(160deg,#f5f3ff,#ede9fe); box-shadow: 8px 8px 0 rgba(124,58,237,0.12);">
                    <div style='font-weight:700;'>文本输入</div>
                    <div style='font-size:0.82rem; color:#334155; margin-top:0.35rem;'>MLLM 诊断 JSON</div>
                </div>
                <div style="display:flex; align-items:center; justify-content:center; font-size:1.35rem; color:#64748b;">→</div>
                <div style="border:1px solid #5eead4; border-radius:8px; padding:0.8rem; background:linear-gradient(160deg,#f0fdfa,#ccfbf1); box-shadow: 8px 8px 0 rgba(15,118,110,0.12);">
                    <div style='font-weight:700;'>属性一致性头</div>
                    <div style='font-size:0.82rem; color:#334155; margin-top:0.35rem;'>type / loc / sev / morph</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_3d_network() -> None:
    render_static_architecture_figure()


def render_static_architecture_figure() -> None:
    def tint(hex_color: str, factor: float) -> str:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        if factor >= 0:
            r = int(r + (255 - r) * factor)
            g = int(g + (255 - g) * factor)
            b = int(b + (255 - b) * factor)
        else:
            scale = 1 + factor
            r = int(r * scale)
            g = int(g * scale)
            b = int(b * scale)
        return f"#{r:02x}{g:02x}{b:02x}"

    def cube(x, y, w, h, color, title, subtitle="", fs=16, ss=11, depth=18):
        front = color
        top = tint(color, 0.28)
        side = tint(color, -0.16)
        stroke = tint(color, -0.42)
        text_lines = [title] + ([subtitle] if subtitle else [])
        label_y = y + h * 0.45
        title_y = label_y - 6 if subtitle else label_y
        subtitle_y = label_y + 18 if subtitle else None
        parts = [
            f'<g filter="url(#shadow)">',
            f'<polygon points="{x},{y} {x+depth},{y-depth} {x+w+depth},{y-depth} {x+w},{y}" fill="{top}" stroke="{stroke}" stroke-width="1"/>',
            f'<polygon points="{x+w},{y} {x+w+depth},{y-depth} {x+w+depth},{y+h-depth} {x+w},{y+h}" fill="{side}" stroke="{stroke}" stroke-width="1"/>',
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{front}" stroke="{stroke}" stroke-width="1.5"/>',
            f'<text x="{x+w/2}" y="{title_y}" text-anchor="middle" font-size="{fs}" font-weight="700" fill="#0f172a">'
            + title.replace("\n", "</text><text x=\"%s\" y=\"%s\" text-anchor=\"middle\" font-size=\"%s\" font-weight=\"700\" fill=\"#0f172a\">" % (x + w/2, title_y + 18, fs))
            + "</text>",
        ]
        if subtitle:
            parts.extend(
                [
                    f'<text x="{x+w/2}" y="{subtitle_y}" text-anchor="middle" font-size="{ss}" fill="#334155">{subtitle}</text>',
                ]
            )
        parts.append("</g>")
        return "".join(parts)

    def header(x, y, w, title, color):
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="42" rx="10" fill="{tint(color, 0.86)}" stroke="{color}" stroke-width="1.2"/>'
            f'<text x="{x+w/2}" y="{y+26}" text-anchor="middle" font-size="14" font-weight="800" fill="{tint(color, -0.45)}">{title}</text>'
        )

    def panel(x, y, w, h):
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#ffffff" stroke="#d8e0ea" stroke-width="1.5"/>'

    def arrow(x1, y1, x2, y2, color="#334155", dashed=False, width=2.2):
        dash = 'stroke-dasharray="8,7"' if dashed else ""
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)" {dash}/>'
        )

    def dot(x, y, color):
        return f'<circle cx="{x}" cy="{y}" r="5.8" fill="{color}"/>'

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 2400 1450">']
    svg.append(
        """
        <defs>
          <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#ffffff"/>
            <stop offset="100%" stop-color="#f8fbff"/>
          </linearGradient>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#334155"/>
          </marker>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#c9d5e4" flood-opacity="0.55"/>
          </filter>
        </defs>
        """
    )
    svg.append('<rect x="0" y="0" width="2400" height="1450" fill="url(#bgGrad)"/>')
    svg.append('<text x="1200" y="62" text-anchor="middle" font-size="36" font-weight="800" fill="#0f172a">AEGIS-Net: 属性级证据一致性验证网络</text>')
    svg.append('<text x="1200" y="96" text-anchor="middle" font-size="16" fill="#64748b">Global Blade Image + Defect Patch + BBox Geometry + MLLM Diagnosis JSON</text>')

    col_x = [24, 319, 614, 909, 1204, 1499, 1794, 2089]
    col_w = 267
    panel_y = 126
    panel_h = 1110
    colors = {
        "blue": "#8fbbe5",
        "blue2": "#a8cff0",
        "orange": "#f4a259",
        "orange2": "#ffc287",
        "purple": "#c7a0ea",
        "purple2": "#d7baf4",
        "green": "#b9d9a8",
        "green2": "#d7e8ca",
        "gray": "#dddddd",
    }
    titles = [
        "1. INPUT LAYER",
        "2. ENCODING LAYER",
        "3. FUSION LAYER",
        "4. ATTRIBUTE QUERY LAYER",
        "5. CROSS-ATTENTION LAYER",
        "6. ATTRIBUTE-LEVEL EVIDENCE",
        "7. PREDICTION HEADS",
        "8. OUTPUT LAYER",
    ]
    for i, x in enumerate(col_x):
        svg.append(panel(x, panel_y, col_w, panel_h))
        svg.append(header(x + 5, panel_y + 7, col_w - 10, titles[i], ["#7da7df", "#6da0dc", "#8d72d8", "#8d72d8", "#8d72d8", "#7396d8", "#7ca56d", "#88b06f"][i]))

    # Column 1: inputs
    input_y = [210, 384, 558, 732]
    input_labels = [
        ("Global Blade Image", "全图", colors["blue2"], colors["blue"]),
        ("Defect Patch", "局部缺陷", colors["orange2"], colors["orange"]),
        ("BBox Geometry", "(x1, y1, x2, y2)", "#e9eef5", "#aab6c8"),
        ("MLLM Diagnosis JSON", "类型 / 位置 / 严重度 / 形态", "#f0f3f8", "#b5c1d1"),
    ]
    for yy, (t, s, c, stroke) in zip(input_y, input_labels):
        svg.append(cube(col_x[0] + 20, yy, 220, 112, c, t, s, fs=15, ss=10, depth=12))
    # Column 2: encoders
    enc_y = [206, 380, 554, 728]
    enc_labels = [
        ("Global Image\nEncoder", "V_global"),
        ("Patch\nEncoder", "V_local"),
        ("BBox Geometry\nEncoder", "G_bbox"),
        ("Text\nEncoder", "T_diag"),
    ]
    enc_colors = [colors["blue"], colors["blue"], colors["blue"], colors["orange"]]
    for yy, (t, s), c in zip(enc_y, enc_labels, enc_colors):
        svg.append(cube(col_x[1] + 22, yy, 210, 108, c, t, s, fs=15, ss=11, depth=14))

    # Column 3: fusion
    svg.append(cube(col_x[2] + 18, 312, 228, 150, "#8ab2e3", "Visual Fusion\nModule", "global + patch + bbox", fs=17, ss=11, depth=16))
    svg.append(cube(col_x[2] + 18, 548, 228, 112, "#d6e9ff", "Visual Evidence\nTokens V", "fused visual evidence", fs=16, ss=10, depth=12))
    svg.append(cube(col_x[2] + 18, 722, 228, 112, "#f5d6af", "Text Tokens T", "diagnosis semantics", fs=16, ss=10, depth=12))

    # Column 4: query bank
    svg.append(cube(col_x[3] + 14, 238, 238, 362, "#cda7ef", "Attribute Query Bank", "q_type / q_loc / q_sev / q_morph", fs=17, ss=11, depth=16))
    qys = [304, 390, 476, 562]
    qlabs = ["q_type", "q_loc", "q_sev", "q_morph"]
    for yy, lab in zip(qys, qlabs):
        svg.append(cube(col_x[3] + 54, yy, 158, 54, "#ead9fa", lab, "learnable query", fs=15, ss=10, depth=10))

    # Column 5: cross attention
    ca_rows = [260, 370, 480, 590]
    for yy, name in zip(ca_rows, ["Type", "Location", "Severity", "Morphology"]):
        svg.append(cube(col_x[4] + 10, yy, 108, 60, "#dbc8f3", f"{name}\nVisual CA", "visual cross-attention", fs=13, ss=9, depth=8))
        svg.append(cube(col_x[4] + 138, yy, 108, 60, "#f1d5a9", f"{name}\nText CA", "text cross-attention", fs=13, ss=9, depth=8))

    # Column 6: attribute-level evidence
    ev_y = [258, 368, 478, 588]
    ev_labs = [
        ("v_type", "t_type"),
        ("v_loc", "t_loc"),
        ("v_sev", "t_sev"),
        ("v_morph", "t_morph"),
    ]
    for yy, (vl, tl) in zip(ev_y, ev_labs):
        svg.append(cube(col_x[5] + 24, yy, 74, 58, "#96b9ef", vl, "visual evidence", fs=14, ss=9, depth=9))
        svg.append(cube(col_x[5] + 154, yy, 74, 58, "#f3c38b", tl, "text evidence", fs=14, ss=9, depth=9))

    # Column 7: heads
    head_y = [228, 332, 436, 540, 644]
    head_labs = [
        ("Type Head", "type_score"),
        ("Location Head", "loc_score"),
        ("Severity Head", "sev_score"),
        ("Morphology Head", "morph_score"),
        ("Hallucination Head", "hallucination_score"),
    ]
    for yy, (t, s) in zip(head_y, head_labs):
        svg.append(cube(col_x[6] + 16, yy, 234, 72, "#cfe2bf", t, s, fs=15, ss=10, depth=12))

    # Column 8: outputs
    out_y = [210, 328, 446, 564, 682]
    out_labs = [
        ("Type Consistency Score", "0.88"),
        ("Location Consistency Score", "0.22"),
        ("Severity Consistency Score", "0.58"),
        ("Morphology Consistency Score", "0.72"),
        ("Overall Hallucination Probability", "0.79"),
    ]
    for yy, (t, s) in zip(out_y, out_labs):
        svg.append(cube(col_x[7] + 18, yy, 220, 78, "#f6fbf3", t, s, fs=13, ss=13, depth=10))
    svg.append(cube(col_x[7] + 18, 812, 220, 88, "#f8f2f2", "Review\nFeedback", "rerun or human review", fs=15, ss=10, depth=10))

    # Connections
    svg.append(arrow(col_x[0] + 240, 266, col_x[1] + 22, 262, "#6ea0da"))
    svg.append(arrow(col_x[0] + 240, 440, col_x[1] + 22, 436, "#6ea0da"))
    svg.append(arrow(col_x[0] + 240, 614, col_x[1] + 22, 610, "#6ea0da"))
    svg.append(arrow(col_x[0] + 240, 788, col_x[1] + 22, 784, "#d89245"))
    svg.append(arrow(col_x[1] + 232, 262, col_x[2] + 18, 374, "#4a83c4"))
    svg.append(arrow(col_x[1] + 232, 436, col_x[2] + 18, 374, "#4a83c4"))
    svg.append(arrow(col_x[1] + 232, 610, col_x[2] + 18, 374, "#4a83c4"))
    svg.append(arrow(col_x[1] + 232, 784, col_x[2] + 18, 784, "#d89245"))
    svg.append(arrow(col_x[2] + 246, 388, col_x[3] + 14, 418, "#7b49bf"))
    svg.append(arrow(col_x[2] + 246, 604, col_x[3] + 14, 418, "#7b49bf"))
    svg.append(arrow(col_x[3] + 252, 364, col_x[4] + 10, 290, "#7b49bf"))
    svg.append(arrow(col_x[3] + 252, 450, col_x[4] + 10, 400, "#7b49bf"))
    svg.append(arrow(col_x[3] + 252, 536, col_x[4] + 10, 510, "#7b49bf"))
    svg.append(arrow(col_x[3] + 252, 622, col_x[4] + 10, 620, "#7b49bf"))
    for y1, y2 in zip([290, 400, 510, 620], [287, 397, 507, 617]):
        svg.append(arrow(col_x[4] + 224, y1, col_x[5] + 24, y2, "#7d67b3"))
        svg.append(arrow(col_x[4] + 224, y1, col_x[5] + 154, y2, "#c68d46"))
    for y1, y2 in zip([287, 397, 507, 617], [259, 369, 479, 589]):
        svg.append(arrow(col_x[5] + 98, y1 + 18, col_x[6] + 16, y2 + 18, "#4f8c52"))
        svg.append(arrow(col_x[5] + 228, y1 + 18, col_x[6] + 16, y2 + 18, "#4f8c52"))
    for y1, y2 in zip([264, 368, 472, 576, 680], [246, 350, 454, 558, 662]):
        svg.append(arrow(col_x[6] + 250, y1, col_x[7] + 18, y2, "#4f8c52"))

    # Evidence labels
    svg.append('<text x="1185" y="214" font-size="12" fill="#4a83c4" font-weight="700">Visual Cross-Attention</text>')
    svg.append('<text x="1328" y="214" font-size="12" fill="#c68d46" font-weight="700">Text Cross-Attention</text>')
    svg.append('<text x="1788" y="214" font-size="12" fill="#4f8c52" font-weight="700">Attribute-level Evidence</text>')

    # Loss strip
    svg.append('<rect x="24" y="1288" width="2350" height="124" rx="16" fill="#f8f8f8" stroke="#d0d7df" stroke-dasharray="10,8" stroke-width="1.5"/>')
    svg.append('<text x="58" y="1355" font-size="20" font-weight="800" fill="#2d3748">Training Losses</text>')
    loss_boxes = [
        (230, "L_attr", "Attribute Supervision", "#ffffff"),
        (570, "L_halluc", "Hallucination Supervision", "#ffffff"),
        (920, "optional L_orth", "Query / Evidence Disentanglement", "#ffffff"),
        (1335, "optional L_ord", "Severity Ordinal Consistency", "#ffffff"),
        (1720, "optional L_match", "Cross-modal Evidence Alignment", "#ffffff"),
    ]
    for x, t, s, c in loss_boxes:
        svg.append(cube(x, 1312, 250, 74, "#efefef", t, s, fs=16, ss=11, depth=10))

    # Loss arrows
    svg.append(arrow(1040, 1210, 355, 1312, "#5a8c61", dashed=True))
    svg.append(arrow(1260, 1210, 695, 1312, "#5a8c61", dashed=True))
    svg.append(arrow(1500, 1210, 1045, 1312, "#7b67b3", dashed=True))
    svg.append(arrow(1720, 1210, 1440, 1312, "#7b67b3", dashed=True))
    svg.append(arrow(1920, 1210, 1860, 1312, "#7b67b3", dashed=True))

    # small connectors / dots
    for x, y, c in [
        (col_x[0] + 240, 266, "#6ea0da"),
        (col_x[0] + 240, 440, "#6ea0da"),
        (col_x[0] + 240, 614, "#6ea0da"),
        (col_x[0] + 240, 788, "#d89245"),
        (col_x[1] + 22, 262, "#6ea0da"),
        (col_x[1] + 22, 436, "#6ea0da"),
        (col_x[1] + 22, 610, "#6ea0da"),
        (col_x[1] + 22, 784, "#d89245"),
        (col_x[7] + 18, 246, "#5ca55d"),
        (col_x[7] + 18, 364, "#5ca55d"),
        (col_x[7] + 18, 482, "#5ca55d"),
        (col_x[7] + 18, 600, "#5ca55d"),
    ]:
        svg.append(dot(x, y, c))

    svg.append("</svg>")
    components.html("".join(svg), height=1450, scrolling=False)


def render_aegis_playground(mode: str = "overview", height: int = 680) -> None:
    html = f"""
    <div id="aegis-playground-{mode}" class="aegis-wrap">
      <div class="topbar">
        <button id="playBtn" class="round">▶</button>
        <button id="stepBtn" class="ghost">单步</button>
        <button id="resetBtn" class="ghost">重置</button>
        <label>Epoch <span id="epoch">000</span></label>
        <label>Learning rate <select id="lr"><option>0.03</option><option selected>0.01</option><option>0.003</option></select></label>
        <label>Mode <select id="mode"><option value="train">Train</option><option value="infer">Infer</option><option value="overview">Overview</option></select></label>
        <label>Scenario <select id="scenario"><option value="wrongLoc">类型可信 / 位置冲突</option><option value="wrongType">类型冲突 / 位置可信</option><option value="clean">诊断整体可信</option></select></label>
      </div>
      <div class="main">
        <aside class="panel left">
          <h3>INPUT</h3>
          <div class="thumb blade"><span>WTBD</span><b>全图</b></div>
          <div class="thumb patch"><span>bbox</span><b>patch</b></div>
          <div class="jsonBox">
            <b>MLLM JSON</b>
            <pre id="jsonText"></pre>
          </div>
          <div class="note">MVP 输入：global image + defect patch + normalized bbox + one MLLM diagnosis JSON.</div>
        </aside>
        <section class="canvasBox">
          <canvas id="net" width="1280" height="560"></canvas>
          <div class="legend">
            <span><i class="blue"></i>视觉特征流</span>
            <span><i class="orange"></i>诊断文本语义</span>
            <span><i class="green"></i>属性一致性证据</span>
            <span><i class="red"></i>属性冲突/幻觉风险</span>
          </div>
        </section>
        <aside class="panel right">
          <h3>OUTPUT</h3>
          <div class="loss">
            <span>Training loss</span><b id="loss">0.940</b>
            <canvas id="lossCanvas" width="220" height="78"></canvas>
          </div>
          <div id="scores" class="scores"></div>
          <div class="decision" id="decision"></div>
        </aside>
      </div>
    </div>
    <style>
      .aegis-wrap {{font-family: Arial, "Microsoft YaHei", sans-serif; border:1px solid #d8e0ea; border-radius:10px; overflow:hidden; background:#f8fafc;}}
      .topbar {{display:flex; align-items:center; gap:16px; padding:12px 16px; background:white; border-bottom:1px solid #e2e8f0; flex-wrap:wrap;}}
      .topbar label {{font-size:13px; color:#475569; display:flex; flex-direction:column; gap:4px;}}
      .topbar select {{border:0; border-bottom:1px solid #cbd5e1; padding:4px 24px 4px 2px; background:white; color:#0f172a;}}
      .round {{width:56px; height:56px; border-radius:50%; border:0; background:#0f4c5c; color:white; font-size:22px; cursor:pointer;}}
      .ghost {{border:1px solid #cbd5e1; background:white; color:#0f172a; border-radius:8px; padding:8px 12px; cursor:pointer;}}
      .main {{display:grid; grid-template-columns: 220px minmax(620px, 1fr) 260px; gap:0; min-height:560px;}}
      .panel {{background:#ffffff; padding:18px; border-right:1px solid #e2e8f0;}}
      .panel.right {{border-right:0; border-left:1px solid #e2e8f0;}}
      .panel h3 {{margin:0 0 14px 0; font-size:15px; letter-spacing:0; color:#0f172a;}}
      .thumb {{height:86px; border:1px solid #cbd5e1; border-radius:8px; margin-bottom:12px; display:flex; align-items:end; justify-content:space-between; padding:10px; color:#0f172a; background:linear-gradient(135deg,#dbeafe,#f8fafc); box-sizing:border-box;}}
      .thumb.patch {{height:72px; background:linear-gradient(135deg,#fee2e2,#fff7ed);}}
      .thumb span {{font-size:12px; color:#64748b;}}
      .thumb b {{font-size:18px;}}
      .jsonBox {{border:1px solid #e2e8f0; border-radius:8px; padding:10px; background:#f8fafc;}}
      .jsonBox pre {{font-size:12px; white-space:pre-wrap; margin:8px 0 0; line-height:1.45; color:#334155;}}
      .note {{font-size:12px; line-height:1.55; color:#64748b; margin-top:12px;}}
      .canvasBox {{position:relative; background:linear-gradient(180deg,#ffffff,#f8fafc); overflow:hidden;}}
      #net {{display:block; width:100%; height:560px;}}
      .legend {{position:absolute; left:18px; bottom:12px; display:flex; gap:12px; font-size:12px; color:#475569; flex-wrap:wrap;}}
      .legend i {{display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:5px; vertical-align:-1px;}}
      .blue {{background:#2b8cbe;}} .orange {{background:#f28e2b;}} .green {{background:#0f766e;}} .red {{background:#dc2626;}}
      .loss {{border:1px solid #e2e8f0; border-radius:8px; padding:10px; background:#f8fafc; margin-bottom:12px;}}
      .loss span {{font-size:12px; color:#64748b;}} .loss b {{display:block; font-size:26px; color:#0f172a; margin-top:4px;}}
      .score {{border:1px solid #e2e8f0; border-radius:8px; padding:9px 10px; margin-bottom:8px; background:white;}}
      .scoreTop {{display:flex; justify-content:space-between; font-size:13px; color:#0f172a; margin-bottom:6px;}}
      .bar {{height:8px; border-radius:8px; background:#e2e8f0; overflow:hidden;}}
      .fill {{height:100%; border-radius:8px; transition:width .25s ease;}}
      .decision {{font-size:13px; line-height:1.55; border-left:4px solid #dc2626; padding:10px; background:#fef2f2; border-radius:6px; color:#7f1d1d;}}
      @media(max-width: 980px) {{.main {{grid-template-columns:1fr;}} .panel.right {{border-left:0; border-top:1px solid #e2e8f0;}} .canvasBox {{min-height:520px;}}}}
    </style>
    <script>
    (function() {{
      const root = document.getElementById("aegis-playground-{mode}");
      const canvas = root.querySelector("#net");
      const ctx = canvas.getContext("2d");
      const lossCanvas = root.querySelector("#lossCanvas");
      const lctx = lossCanvas.getContext("2d");
      const playBtn = root.querySelector("#playBtn");
      const stepBtn = root.querySelector("#stepBtn");
      const resetBtn = root.querySelector("#resetBtn");
      const epochEl = root.querySelector("#epoch");
      const lossEl = root.querySelector("#loss");
      const scoresEl = root.querySelector("#scores");
      const decisionEl = root.querySelector("#decision");
      const scenarioEl = root.querySelector("#scenario");
      const modeEl = root.querySelector("#mode");
      const jsonEl = root.querySelector("#jsonText");
      modeEl.value = "{mode}" === "training" ? "train" : ("{mode}" === "inference" ? "infer" : "overview");

      let epoch = 0, running = modeEl.value !== "overview", timer = null;
      let lossHistory = [];
      const attrs = ["Type", "Location", "Severity", "Morphology"];
      const colors = {{vision:"#2b8cbe", text:"#f28e2b", good:"#0f766e", bad:"#dc2626", neutral:"#94a3b8"}};
      const scenarios = {{
        wrongLoc: {{json:{{类型:"雷击损伤", 位置:"图像右下区域", 严重度:"严重", 形态:"黑色烧蚀斑块"}}, target:{{Type:.88, Location:.22, Severity:.58, Morphology:.72}}, halluc:.79}},
        wrongType: {{json:{{类型:"腐蚀", 位置:"图像右中区域", 严重度:"中度", 形态:"局部材料剥离"}}, target:{{Type:.18, Location:.86, Severity:.62, Morphology:.68}}, halluc:.74}},
        clean: {{json:{{类型:"雷击损伤", 位置:"图像右中区域", 严重度:"中度", 形态:"深色烧蚀并伴随边缘裂纹"}}, target:{{Type:.91, Location:.84, Severity:.66, Morphology:.79}}, halluc:.18}}
      }};

      function node(x,y,label,kind,active=false) {{ return {{x,y,label,kind,active}}; }}
      const nodes = [
        node(.06,.20,"Global Image","input"), node(.06,.43,"Defect Patch","input"), node(.06,.66,"BBox Geometry","input"), node(.06,.84,"MLLM JSON","input"),
        node(.19,.20,"Global Encoder","conv"), node(.19,.43,"Patch Encoder","conv"), node(.19,.66,"BBox MLP","mlp"), node(.19,.84,"Text Encoder","text"),
        node(.36,.28,"Visual Tokens","token"), node(.36,.66,"Text Tokens","token"),
        node(.50,.47,"Cross Attention","fusion"),
        node(.62,.25,"Type Query","query"), node(.62,.41,"Location Query","query"), node(.62,.57,"Severity Query","query"), node(.62,.73,"Morphology Query","query"),
        node(.76,.25,"Type Head","head"), node(.76,.41,"Location Head","head"), node(.76,.57,"Severity Head","head"), node(.76,.73,"Morphology Head","head"),
        node(.90,.50,"Hallucination Head","out")
      ];

      function scenario() {{ return scenarios[scenarioEl.value]; }}
      function scores() {{
        const s = scenario();
        let warm = Math.min(1, epoch / 80);
        if (modeEl.value === "infer") warm = Math.min(1, epoch / 24);
        const out = {{}};
        attrs.forEach(a => {{
          const base = modeEl.value === "train" ? .5 + Math.sin((epoch + a.length) / 9) * .12 : .42;
          out[a] = base + (s.target[a] - base) * warm;
          out[a] = Math.max(.05, Math.min(.97, out[a]));
        }});
        return out;
      }}
      function currentLoss() {{
        const s = scores();
        const target = scenario().target;
        let err = 0;
        attrs.forEach(a => err += Math.abs(s[a] - target[a]));
        return Math.max(.03, err / attrs.length + (modeEl.value === "train" ? .08 * Math.exp(-epoch/40) : .02));
      }}
      function drawLayerBlock(x,y,w,h,depth,label,color) {{
        for(let i=depth;i>=0;i--) {{
          ctx.fillStyle = i===0 ? color : "rgba(253,186,116,.32)";
          ctx.strokeStyle = "rgba(154,52,18,.32)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x+i*4,y-i*4); ctx.lineTo(x+w+i*4,y-i*4); ctx.lineTo(x+w+i*4,y+h-i*4); ctx.lineTo(x+i*4,y+h-i*4); ctx.closePath();
          ctx.fill(); ctx.stroke();
        }}
        ctx.fillStyle="#334155"; ctx.font="12px Arial"; ctx.fillText(label,x,y+h+18);
      }}
      function drawConnection(a,b,weight,kind) {{
        const t = (epoch/12) % 1;
        ctx.save();
        ctx.lineWidth = 1 + weight * 4;
        ctx.strokeStyle = kind === "bad" ? "rgba(220,38,38,.65)" : kind === "text" ? "rgba(242,142,43,.55)" : kind === "good" ? "rgba(15,118,110,.65)" : "rgba(43,140,190,.55)";
        ctx.beginPath();
        const cx = (a.x+b.x)/2;
        ctx.moveTo(a.x,a.y);
        ctx.bezierCurveTo(cx,a.y,cx,b.y,b.x,b.y);
        ctx.stroke();
        const px = a.x + (b.x-a.x)*t, py = a.y + (b.y-a.y)*t;
        ctx.fillStyle = kind === "bad" ? colors.bad : kind === "text" ? colors.text : colors.vision;
        ctx.beginPath(); ctx.arc(px,py,4+weight*2,0,Math.PI*2); ctx.fill();
        ctx.restore();
      }}
      function drawNeuron(n, score=.5) {{
        const x=n.x*canvas.width, y=n.y*canvas.height;
        if(n.kind === "conv") {{ drawLayerBlock(x-18,y-48,34,96,7,n.label,"#fdba74"); return; }}
        if(n.kind === "mlp") {{ drawLayerBlock(x-24,y-26,48,52,3,n.label,"#c4b5fd"); return; }}
        if(n.kind === "token") {{ drawLayerBlock(x-26,y-36,52,72,4,n.label,"#bfdbfe"); return; }}
        let r = n.kind === "query" ? 24 : n.kind === "out" ? 31 : 22;
        ctx.fillStyle = n.kind === "input" ? "#e0f2fe" : n.kind === "text" ? "#ffedd5" : n.kind === "head" ? "#f8fafc" : n.kind === "out" ? "#fee2e2" : "#fef9c3";
        ctx.strokeStyle = score < .35 && n.kind === "head" ? colors.bad : score > .7 && n.kind === "head" ? colors.good : "#334155";
        ctx.lineWidth = n.kind === "query" ? 3 : 2;
        ctx.beginPath(); ctx.roundRect(x-r,y-r,r*2,r*2,8); ctx.fill(); ctx.stroke();
        if(n.kind === "head") {{
          ctx.fillStyle = score < .35 ? colors.bad : score > .7 ? colors.good : "#ca8a04";
          ctx.fillRect(x-r+5,y+r-10,(r*2-10)*score,5);
        }}
        ctx.fillStyle="#0f172a"; ctx.font="12px Arial"; ctx.textAlign="center";
        const parts = n.label.split(" ");
        parts.forEach((p,i)=>ctx.fillText(p,x,y-8+i*13));
        ctx.textAlign="left";
      }}
      function drawOutputSurface(s) {{
        const x=canvas.width*.82, y=canvas.height*.12, w=canvas.width*.14, h=canvas.height*.22;
        ctx.save();
        ctx.fillStyle="#f8fafc"; ctx.strokeStyle="#cbd5e1"; ctx.strokeRect(x,y,w,h);
        for(let i=0;i<22;i++) for(let j=0;j<14;j++) {{
          const val = Math.sin(i*.7 + epoch*.08) + Math.cos(j*.9 + epoch*.05);
          const risky = s.Location < .35 || s.Type < .35;
          ctx.fillStyle = val + (risky ? .25 : -.25) > 0 ? "rgba(242,142,43,.55)" : "rgba(43,140,190,.58)";
          ctx.fillRect(x+i*w/22,y+j*h/14,w/22+1,h/14+1);
        }}
        ctx.fillStyle="#0f172a"; ctx.font="12px Arial"; ctx.fillText("attribute decision map",x,y+h+18);
        ctx.restore();
      }}
      function drawLoss() {{
        lctx.clearRect(0,0,lossCanvas.width,lossCanvas.height);
        lctx.strokeStyle="#cbd5e1"; lctx.strokeRect(0,0,lossCanvas.width,lossCanvas.height);
        lctx.beginPath(); lctx.strokeStyle="#0f172a"; lctx.lineWidth=2;
        lossHistory.forEach((v,i)=> {{
          const x = i/(Math.max(1,lossHistory.length-1))*lossCanvas.width;
          const y = lossCanvas.height - Math.min(.98,v)*lossCanvas.height;
          if(i===0) lctx.moveTo(x,y); else lctx.lineTo(x,y);
        }});
        lctx.stroke();
      }}
      function updateSidePanels(s, loss) {{
        epochEl.textContent = String(epoch).padStart(3,"0");
        lossEl.textContent = loss.toFixed(3);
        jsonEl.textContent = JSON.stringify(scenario().json, null, 2);
        scoresEl.innerHTML = attrs.map(a => {{
          const v = s[a], color = v < .35 ? colors.bad : v > .7 ? colors.good : "#ca8a04";
          return `<div class="score"><div class="scoreTop"><b>${{a}}</b><span>${{v.toFixed(2)}}</span></div><div class="bar"><div class="fill" style="width:${{v*100}}%; background:${{color}}"></div></div></div>`;
        }}).join("");
        const risky = s.Location < .35 || s.Type < .35;
        const halluc = scenario().halluc * Math.min(1, .35 + epoch/60);
        decisionEl.innerHTML = risky ? `复核建议：关键属性存在冲突。<br>整体幻觉概率约 <b>${{halluc.toFixed(2)}}</b>。优先检查类型/位置描述。`
                                   : `复核建议：当前属性基本被图像证据支持。<br>整体幻觉概率约 <b>${{halluc.toFixed(2)}}</b>。`;
      }}
      function draw() {{
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.max(900, rect.width * window.devicePixelRatio);
        canvas.height = 560 * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        const W = rect.width, H = 560;
        ctx.clearRect(0,0,W,H);
        ctx.fillStyle="#ffffff"; ctx.fillRect(0,0,W,H);
        const s = scores();
        const N = nodes.map(n => ({{...n, x:n.x*W, y:n.y*H}}));
        const by = Object.fromEntries(N.map(n=>[n.label,n]));
        [
          ["Global Image","Global Encoder","vision"],
          ["Defect Patch","Patch Encoder","vision"],
          ["BBox Geometry","BBox MLP","vision"],
          ["MLLM JSON","Text Encoder","text"],
          ["Global Encoder","Visual Tokens","vision"],
          ["Patch Encoder","Visual Tokens","vision"],
          ["BBox MLP","Visual Tokens","vision"],
          ["Text Encoder","Text Tokens","text"],
          ["Visual Tokens","Cross Attention","vision"],
          ["Text Tokens","Cross Attention","text"]
        ].forEach((e,i)=>drawConnection(by[e[0]],by[e[1]],.35+.12*Math.sin(epoch/10+i),e[2]));
        attrs.forEach((a,idx)=> {{
          const q = by[a==="Location"?"Location Query":a==="Severity"?"Severity Query":a==="Morphology"?"Morphology Query":"Type Query"];
          const h = by[a==="Location"?"Location Head":a==="Severity"?"Severity Head":a==="Morphology"?"Morphology Head":"Type Head"];
          const kind = s[a] < .35 ? "bad" : s[a] > .7 ? "good" : "vision";
          drawConnection(by["Cross Attention"],q,.35+s[a]*.4,kind);
          drawConnection(q,h,.25+s[a]*.55,kind);
          drawConnection(h,by["Hallucination Head"],.25+(1-s[a])*.55,s[a] < .35 ? "bad" : "good");
        }});
        N.forEach(n => {{
          const key = n.label.includes("Type") ? "Type" : n.label.includes("Location") ? "Location" : n.label.includes("Severity") ? "Severity" : n.label.includes("Morphology") ? "Morphology" : null;
          drawNeuron(n, key ? s[key] : .5);
        }});
        drawOutputSurface(s);
        const loss = currentLoss();
        if(lossHistory.length === 0 || epoch % 2 === 0) lossHistory.push(loss);
        if(lossHistory.length > 90) lossHistory.shift();
        drawLoss();
        updateSidePanels(s, loss);
      }}
      function tick() {{ epoch += modeEl.value === "infer" ? 2 : 1; draw(); }}
      function play() {{
        running = !running;
        playBtn.textContent = running ? "Ⅱ" : "▶";
        if(running) timer = setInterval(tick, modeEl.value === "infer" ? 170 : 120);
        else clearInterval(timer);
      }}
      playBtn.onclick = play;
      stepBtn.onclick = () => {{ tick(); }};
      resetBtn.onclick = () => {{ epoch=0; lossHistory=[]; draw(); }};
      scenarioEl.onchange = () => {{ epoch=0; lossHistory=[]; draw(); }};
      modeEl.onchange = () => {{ epoch=0; lossHistory=[]; draw(); }};
      if (modeEl.value !== "overview") {{ playBtn.textContent = "Ⅱ"; timer = setInterval(tick, modeEl.value === "infer" ? 170 : 120); }}
      if (!CanvasRenderingContext2D.prototype.roundRect) {{
        CanvasRenderingContext2D.prototype.roundRect = function(x,y,w,h,r) {{ this.moveTo(x+r,y); this.arcTo(x+w,y,x+w,y+h,r); this.arcTo(x+w,y+h,x,y+h,r); this.arcTo(x,y+h,x,y,r); this.arcTo(x,y,x+w,y,r); return this; }}
      }}
      draw();
      window.addEventListener("resize", draw);
    }})();
    </script>
    """
    components.html(html, height=height, scrolling=False)


def model_structure_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "模块": "Global Image Encoder",
                "输入": "global_images: [B, 3, 512, 512]",
                "输出": "V_global",
                "小白理解": "先看整张图，知道缺陷在叶片整体中的位置和背景。",
            },
            {
                "模块": "Patch Encoder",
                "输入": "patch_images: [B, 3, 224, 224]",
                "输出": "V_local",
                "小白理解": "再放大看缺陷局部，捕捉裂纹、烧蚀、腐蚀等细节。",
            },
            {
                "模块": "BBox Geometry MLP",
                "输入": "bbox_features: [B, 4]",
                "输出": "G_bbox",
                "小白理解": "把归一化框坐标编码进去，帮助模型判断文本里说的左下、右中是否靠谱。",
            },
            {
                "模块": "Text Encoder",
                "输入": "input_ids / attention_mask: [B, max_len]",
                "输出": "T_diag",
                "小白理解": "把诊断报告里的类型、位置、严重度、形态变成模型能比较的语义向量。",
            },
            {
                "模块": "Cross-Attention Fusion",
                "输入": "V_global + V_local + G_bbox + T_diag",
                "输出": "F_cross",
                "小白理解": "让图像证据和诊断语义互相对齐，找出文本描述是否有图像支撑。",
            },
            {
                "模块": "Attribute Queries",
                "输入": "F_cross",
                "输出": "E_type / E_loc / E_sev / E_morph",
                "小白理解": "四个专门问题分别去找证据：类型对不对，位置对不对，严重度和形态是否有图像支撑。",
            },
            {
                "模块": "Consistency Heads",
                "输入": "E_type / E_loc / E_sev / E_morph",
                "输出": "type_score / loc_score / sev_score / morph_score",
                "小白理解": "逐项打分，不直接替代 MLLM，而是审查 MLLM 有没有说错。",
            },
            {
                "模块": "Hallucination Head",
                "输入": "四个属性分数 + 属性证据",
                "输出": "hallucination_score",
                "小白理解": "只要关键属性冲突明显，整条诊断就应该进入复核队列。",
            },
        ]
    )


def render_route_overview() -> None:
    st.subheader("路线总览")
    st.caption("这一页只回答一个问题：当前原型和未来 AEGIS-Net 是什么关系。")
    comparison = pd.DataFrame(
        [
            {"维度": "核心目标", "页面 1：当前原型": "验证 MLLM 文本中类型与位置是否和 WTBD 标注一致", "未来 AEGIS-Net": "学习图像证据与诊断文本属性之间的一致性"},
            {"维度": "输入", "页面 1：当前原型": "图像、XML 类别、bbox、MLLM JSON", "未来 AEGIS-Net": "global image、patch、bbox、MLLM JSON"},
            {"维度": "验证方式", "页面 1：当前原型": "类型查表 + bbox 九宫格规则", "未来 AEGIS-Net": "视觉/文本编码器 + Cross-Attention + Attribute Query"},
            {"维度": "输出", "页面 1：当前原型": "一致/不一致/暂未启用", "未来 AEGIS-Net": "属性一致性分数 + 整体幻觉概率"},
            {"维度": "是否依赖测试真值", "页面 1：当前原型": "依赖 XML 作参考", "未来 AEGIS-Net": "正式推理不输入 XML 真值"},
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.info("最关键的定位：AEGIS-Net 不是输入图片后生成诊断文字，而是输入图片 + MLLM 诊断文本，输出每个诊断属性是否可信。")


def render_data_contract_page() -> None:
    st.subheader("数据集、输入与输出格式")
    st.caption("这一页说明训练数据一条样本长什么样，以及模型真正吃进去和吐出来的是什么。")

    st.markdown("**样本单位：一个缺陷实例**")
    c1, c2, c3 = st.columns(3)
    with c1:
        section_card("图像级输入", "一张原始全图 + 一个 bbox。若一张图有 3 个 bbox，则拆成 3 条缺陷实例样本。", "#0284c7")
    with c2:
        section_card("实例级输入", "每个 bbox 裁剪出一个 patch，并保存 bbox 归一化坐标。", "#0f766e")
    with c3:
        section_card("文本级输入", "MLLM 诊断 JSON 转成一段文本序列，不引入候选描述库作为 MVP 输入。", "#7c3aed")

    st.markdown("**MVP jsonl 一行样本**")
    st.code(
        json.dumps(
            {
                "sample_id": "WTBD_000123_obj0",
                "image_path": "images/000123.jpg",
                "bbox": [1200, 850, 1550, 1030],
                "patch_path": "patches/WTBD_000123_obj0.jpg",
                "gt_reference": {
                    "type": "corrosion",
                    "type_zh": "腐蚀",
                    "image_region": "图像右上区域",
                    "severity": None,
                    "morphology": None,
                },
                "mllm_diagnosis": {
                    "类型": "雷击损伤",
                    "位置": "图像右上区域",
                    "严重度": "严重",
                    "形态": "深色烧蚀并伴随材料剥离",
                },
                "attribute_labels": {
                    "type": 0,
                    "loc": 1,
                    "sev": -1,
                    "morph": -1,
                    "overall_hallucination": 1,
                },
                "label_mask": {"type": 1, "loc": 1, "sev": 0, "morph": 0},
                "split": "train",
            },
            ensure_ascii=False,
            indent=2,
        ),
        language="json",
    )

    st.markdown("**一个 batch 进入模型时的形状**")
    shape_df = pd.DataFrame(
        [
            {"字段": "patch_images", "形状": "[B, 3, 224, 224]", "用途": "局部缺陷视觉证据"},
            {"字段": "global_images", "形状": "[B, 3, 512, 512]", "用途": "全局叶片结构与相对位置背景"},
            {"字段": "bbox_features", "形状": "[B, 4]", "用途": "[xmin/W, ymin/H, xmax/W, ymax/H]"},
            {"字段": "input_ids", "形状": "[B, max_len]", "用途": "诊断 JSON 转文本后的 token"},
            {"字段": "attribute_labels", "形状": "dict of [B]", "用途": "type/loc/sev/morph/halluc 监督信号"},
            {"字段": "label_mask", "形状": "dict of [B]", "用途": "没有标签的属性不计算损失"},
        ]
    )
    st.dataframe(shape_df, use_container_width=True, hide_index=True)

    st.markdown("**模型输出，不是自然语言**")
    st.code(
        json.dumps(
            {
                "type_score": 0.88,
                "loc_score": 0.21,
                "sev_score": None,
                "morph_score": None,
                "hallucination_score": 0.79,
            },
            ensure_ascii=False,
            indent=2,
        ),
        language="json",
    )
    st.warning("`gt_reference` 只用于训练前构造标签；正式推理时不能输入 XML 真值。")


def get_reference_sample(preferred_image_id: str = "10.jpg"):
    if not dataset_exists():
        return None

    image_ids = list_image_ids()
    if not image_ids:
        return None

    image_id = preferred_image_id if preferred_image_id in image_ids else image_ids[0]
    annotation = parse_annotation(image_id)
    if not annotation.objects:
        return None

    image = load_image(annotation)
    defect = annotation.objects[0]
    location_info = calculate_relative_location(defect.bbox, annotation.width, annotation.height)
    return image_id, annotation, image, defect, location_info


def render_verification_page() -> None:
    st.title("AEGIS Demo v0.2：属性级诊断可靠性验证")
    st.caption("页面 1 | 基于 WTBD 真实图像、XML 标签和 bbox 几何计算的原型系统")

    st.info("MLLM 负责生成诊断，AEGIS 负责验证诊断；本页展示当前已经实现的属性级验证流程。")

    with st.expander("当前原型边界", expanded=False):
        st.markdown(
            """
            - 类型验证：使用 WTBD XML 专家类别标注作为参考证据。
            - 位置验证：使用 bbox 中心点计算图像九宫格相对位置。
            - 严重度、形态：WTBD 暂无属性级真值，本版本不输出自动判断。
            - 正式 AEGIS-Net：后续将学习图像证据与诊断文本属性之间的一致性，不依赖测试标签。
            """
        )

    if not dataset_exists():
        st.error(f"未找到 WTBD 数据集目录：{DATASET_DIR}")
        st.stop()

    image_ids = list_image_ids()
    split = read_split()

    st.sidebar.header("数据与案例")
    subset_options = ["all", "train", "val", "test"]
    subset = st.sidebar.selectbox("数据划分", subset_options, index=0)
    filtered_ids = [i for i in image_ids if subset == "all" or split.get(i) == subset]

    default_index = filtered_ids.index("10.jpg") if "10.jpg" in filtered_ids else 0
    image_id = st.sidebar.selectbox("图像编号", filtered_ids, index=default_index)
    annotation = parse_annotation(image_id)
    image = load_image(annotation)

    if not annotation.objects:
        st.warning("该图像没有解析到 object 标注。")
        st.stop()

    object_options = [
        f"目标 {obj.index}: {class_to_zh(obj.class_name)} ({obj.class_name}) "
        f"[{obj.bbox.xmin},{obj.bbox.ymin},{obj.bbox.xmax},{obj.bbox.ymax}]"
        for obj in annotation.objects
    ]
    obj_label = st.sidebar.selectbox("目标选择", object_options)
    obj_index = object_options.index(obj_label)
    defect = annotation.objects[obj_index]
    location_info = calculate_relative_location(defect.bbox, annotation.width, annotation.height)

    cases = load_cases()
    case_key = st.sidebar.radio(
        "预设诊断案例",
        list(cases.keys()),
        format_func=lambda k: cases[k]["label"],
    )
    case_payload = materialize_case(cases[case_key], defect.class_name, location_info["location"])

    if "last_case_key" not in st.session_state or st.session_state.last_case_key != (image_id, obj_index, case_key):
        st.session_state.diagnosis_text = json.dumps(case_payload, ensure_ascii=False, indent=2)
        st.session_state.last_case_key = (image_id, obj_index, case_key)

    st.sidebar.caption(cases[case_key]["description"])

    st.markdown(
        "#### 演示流程\n"
        "`WTBD 叶片图像 → 缺陷 bbox 与 patch → MLLM 结构化诊断 → AEGIS 属性级核查 → 错误属性定位与复核反馈`"
    )

    col1, col2, col3 = st.columns([1.1, 0.9, 1.0])

    with col1:
        st.subheader("1. WTBD 原始图像与 XML 框")
        st.image(
            draw_bbox(image, annotation.objects, selected_index=defect.index),
            caption=f"{annotation.image_id} | subset={split.get(annotation.image_id, 'unknown')} | {annotation.width}×{annotation.height}",
            use_container_width=True,
        )
        st.write(f"XML 真实类别：**{class_to_zh(defect.class_name)}** (`{defect.class_name}`)")
        st.write(f"XML 文件：`{annotation.xml_path.name}`")

    with col2:
        st.subheader("2. 缺陷 patch 与几何证据")
        st.image(crop_patch(image, defect.bbox), caption="bbox 裁剪 patch", use_container_width=True)
        st.image(
            draw_location_grid(image, defect.bbox, location_info["location"]),
            caption=f"九宫格位置：{location_info['location']}",
            use_container_width=True,
        )
        st.markdown(
            f"""
            **bbox 坐标**：`({defect.bbox.xmin}, {defect.bbox.ymin}, {defect.bbox.xmax}, {defect.bbox.ymax})`  
            **中心点**：`({location_info['center_x']:.1f}, {location_info['center_y']:.1f})`  
            **相对位置**：`rx={location_info['rx']:.2f}, ry={location_info['ry']:.2f}`
            """
        )

    with col3:
        st.subheader("3. 可编辑 MLLM 诊断 JSON")
        diagnosis_text = st.text_area(
            "现场可修改类型和位置",
            key="diagnosis_text",
            height=270,
        )
        run = st.button("执行 AEGIS 验证", type="primary", use_container_width=True)

        if run:
            try:
                diagnosis = json.loads(diagnosis_text)
            except json.JSONDecodeError as exc:
                st.error(f"JSON 解析失败：{exc}")
                st.stop()

            results = verify_diagnosis(
                diagnosis,
                defect.class_name,
                defect.bbox,
                annotation.width,
                annotation.height,
            )
            st.session_state.results = results

    st.divider()
    st.subheader("4. AEGIS 属性级验证结果")

    results = st.session_state.get("results")
    if results is None:
        st.caption("点击“执行 AEGIS 验证”后显示结果。")
        st.stop()

    df = result_dataframe(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    status = overall_status(results)
    if "需要复核" in status:
        st.error(status)
    else:
        st.success(status)

    st.text_area("自动反馈", generate_feedback(results), height=210)

    with st.expander("类别映射与可识别位置"):
        st.write("类别映射：")
        st.json(CLASS_MAP)
        st.write("位置枚举：")
        st.write(", ".join(GRID_LABELS))


def render_architecture_overview() -> None:
    st.subheader("模型结构：AEGIS-Net 用什么网络")
    st.caption("这一页只展示静态结构图，方便开题和答辩时讲清楚网络长什么样。")

    tabs = st.tabs(["动态图", "模块表"])
    with tabs[0]:
        draw_3d_network()
    with tabs[1]:
        st.dataframe(model_structure_dataframe(), use_container_width=True, hide_index=True)
        st.info("图中每条分支都有含义：global image / patch / bbox / JSON 进入编码层，再经过融合、属性查询、交叉注意力、证据层、预测头和输出层。")


def render_training_process() -> None:
    st.subheader("训练过程：属性一致性如何被学出来")
    st.caption("这里只保留训练动态和样本格式，不再重复结构总图。")

    sample = get_reference_sample()
    show_sample = st.toggle("查看训练样本", value=True)
    show_queries = st.toggle("查看属性 query 关注点", value=True)

    sample_payload = {
        "type_label": 0,
        "loc_label": 1,
        "sev_label": 1,
        "morph_label": 1,
    }

    top_col, mid_col = st.columns([1.1, 0.9])
    with top_col:
        draw_block_flow("训练阶段数据流")
    with mid_col:
        st.markdown("**训练样本目标**")
        st.json(
            {
                "image": "10.jpg",
                "bbox": "[xmin, ymin, xmax, ymax]",
                "xml_truth": "thunderstrike / 雷击损伤",
                "mllm_diagnosis": {
                    "类型": "腐蚀",
                    "位置": "图像右中区域",
                    "严重度": "严重",
                    "形态": "深色烧蚀并伴随局部材料剥离",
                },
                "attribute_labels": sample_payload,
            }
        )

    st.markdown("**动态训练循环**")
    st.caption("点击播放键，可以看到 epoch 增长、loss 下降、属性 head 分数向标签靠近。当前 MVP 第一版重点训练 type 和 loc。")
    render_aegis_playground(mode="training", height=650)

    if show_sample and sample is not None:
        _, annotation, image, defect, location_info = sample
        diagnosis = {
            "类型": class_to_zh(defect.class_name),
            "位置": location_info["location"],
            "严重度": "中度",
            "形态": "沿叶片表面呈细长线状扩展",
        }
        c1, c2, c3 = st.columns([1.1, 0.9, 1.0])
        with c1:
            st.image(
                draw_bbox(image, annotation.objects, selected_index=defect.index),
                caption="训练样本图像与候选缺陷框",
                use_container_width=True,
            )
        with c2:
            st.image(crop_patch(image, defect.bbox), caption="局部 patch", use_container_width=True)
        with c3:
            st.json(
                {
                    "visual_evidence": {
                        "bbox": defect.bbox.as_tuple(),
                        "xml_class": defect.class_name,
                        "grid_location": location_info["location"],
                    },
                    "mllm_diagnosis": diagnosis,
                    "attribute_labels": {
                        "type_consistent": 1,
                        "location_consistent": 1,
                        "severity_consistent": "expert/rule label",
                        "morphology_consistent": "expert/rule label",
                        "overall_hallucination": 0,
                    },
                }
            )
    elif show_sample:
        st.warning("当前未找到可展示的本地 WTBD 样本，但架构与训练流程仍可查看。")

    if show_queries:
        query_df = pd.DataFrame(
            [
                {"属性 query": "Type Query", "视觉关注点": "纹理、颜色、烧蚀痕迹、裂纹模式", "文本关注点": "缺陷类别词与同义表达"},
                {"属性 query": "Location Query", "视觉关注点": "bbox、全局九宫格、叶片区域", "文本关注点": "左/右/上/下/中心等空间描述"},
                {"属性 query": "Severity Query", "视觉关注点": "面积、长度、范围、密集程度", "文本关注点": "轻微/中度/严重等程度词"},
                {"属性 query": "Morph Query", "视觉关注点": "边界、方向、形状、连续性", "文本关注点": "线状、片状、网状、烧蚀等形态词"},
            ]
        )
        st.dataframe(query_df, use_container_width=True, hide_index=True)

    st.markdown("**训练目标设计**")
    loss_df = pd.DataFrame(
        [
            {"损失项": "L_attr", "作用": "监督每个属性的一致/不一致判断", "v0.2 状态": "MVP 优先"},
            {"损失项": "L_halluc", "作用": "监督整体诊断幻觉概率", "v0.2 状态": "MVP 优先"},
            {"损失项": "L_orth", "作用": "约束不同属性 query 学到互补证据", "v0.2 状态": "可选增强"},
            {"损失项": "L_ord", "作用": "建模严重度轻重顺序关系", "v0.2 状态": "后续扩展"},
            {"损失项": "L_match", "作用": "拉近匹配图文属性、拉远不匹配属性", "v0.2 状态": "后续扩展"},
        ]
    )
    st.dataframe(loss_df, use_container_width=True, hide_index=True)

    st.markdown("**训练反馈摘要**")
    st.write("训练阶段重点是让模型学会：图像证据支持哪些属性，哪些属性需要复核。")
    st.write("对于小白来说，可以把它理解成：模型先看图，再看诊断，再一项一项对答案。")


def render_inference_process() -> None:
    st.subheader("推理过程：只看输入，不看真值")
    st.caption("这里只保留推理动态和输出结果，不再重复训练部分。")

    default_json = {
        "类型": "雷击损伤",
        "位置": "图像右下区域",
        "严重度": "严重",
        "形态": "黑色烧蚀斑块并伴随边缘裂纹",
    }
    st.markdown("**动态推理演示：只看输入，不看答案**")
    st.caption("切换 Scenario 可以看到不同诊断错误如何改变属性分数、连线颜色和整体幻觉概率。")
    render_aegis_playground(mode="inference", height=650)

    col1, col2 = st.columns([0.9, 1.1])
    with col1:
        st.markdown("**待验证 MLLM 诊断样例**")
        st.json(default_json)
    with col2:
        output = pd.DataFrame(
            [
                {"属性": "类型", "一致性分数": 0.88, "判断": "较可信", "反馈": "视觉证据与雷击烧蚀描述较匹配"},
                {"属性": "位置", "一致性分数": 0.24, "判断": "需复核", "反馈": "文本位置与视觉区域证据冲突"},
                {"属性": "严重度", "一致性分数": 0.57, "判断": "不确定", "反馈": "程度词证据不足，需要人工或规则校准"},
                {"属性": "形态", "一致性分数": 0.74, "判断": "较可信", "反馈": "形态描述与局部纹理大体一致"},
            ]
        )
        st.markdown("**目标输出格式示意**")
        st.dataframe(output, use_container_width=True, hide_index=True)
        metric_card("诊断幻觉概率", "0.79", "由于位置属性显著冲突，建议进入人工复核队列。", "#dc2626")
    st.info("推理演示强调的是：AEGIS-Net 不重新替代 MLLM 诊断，而是判断每个诊断属性是否被图像证据支持。")


def render_aegis_net_page() -> None:
    st.title("AEGIS Demo v0.2：未来正式模型展示")
    st.caption("页面 2 | AEGIS-Net 的数据格式、模型结构、训练与推理设计")

    st.warning(
        "本页用于展示未来 AEGIS-Net 的系统设计。当前页面 1 已实现规则/标签辅助验证；页面 2 的分数为示意输出。"
    )

    overview_tab, data_tab, model_tab, train_tab, infer_tab = st.tabs(
        ["路线总览", "数据/输入输出", "模型结构", "训练过程", "推理过程"]
    )
    with overview_tab:
        render_route_overview()
    with data_tab:
        render_data_contract_page()
    with model_tab:
        render_architecture_overview()
    with train_tab:
        render_training_process()
    with infer_tab:
        render_inference_process()

    st.info(
        "边界说明：当前 demo 展示概念与模型设计。页面 1 使用 XML 与 bbox 规则完成可运行验证；页面 2 是未来模型蓝图，示意分数不能作为真实性能指标。"
    )


def main():
    st.set_page_config(
        page_title="AEGIS Demo v0.2",
        page_icon="🛠️",
        layout="wide",
    )

    st.sidebar.title("AEGIS Demo v0.2")
    page = st.sidebar.radio(
        "展示页面",
        ["页面 1：属性级验证原型", "页面 2：AEGIS-Net 训练与推理"],
    )

    if page.startswith("页面 1"):
        render_verification_page()
    else:
        render_aegis_net_page()


if __name__ == "__main__":
    main()
