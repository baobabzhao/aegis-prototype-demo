from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from PIL import ImageDraw

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Streamlit is not installed.") from exc

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


def main():
    st.set_page_config(page_title="AEGIS Prototype v0.1", page_icon="🛠️", layout="wide")
    st.title("AEGIS-Prototype v0.1：属性级诊断可靠性验证演示系统")
    st.caption("基于 WTBD 真实图像、真实 XML 标签和 bbox 几何计算的云端演示版")

    st.info("MLLM 负责生成诊断，AEGIS 负责验证诊断；本原型不直接替代 MLLM 输出最终结论。")

    with st.expander("原型边界", expanded=False):
        st.markdown(
            """
            - 类型验证：使用 WTBD XML 专家类别标注作为参考证据。
            - 位置验证：使用 bbox 中心点计算图像九宫格相对位置。
            - 严重度、形态：当前 WTBD 数据集缺少属性级真值，本版本不输出自动判断。
            - 正式 AEGIS-Net：后续将学习图像证据与诊断文本属性之间的一致性，使测试阶段不依赖真值标签。
            """
        )

    if not dataset_exists():
        st.error(f"未找到云端样本数据目录：{DATASET_DIR}")
        st.stop()

    image_ids = list_image_ids()
    split = read_split()

    st.sidebar.header("数据与案例")
    subset = st.sidebar.selectbox("数据划分", ["all", "train", "val", "test"], index=0)
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
    case_key = st.sidebar.radio("预设诊断案例", list(cases.keys()), format_func=lambda k: cases[k]["label"])
    case_payload = materialize_case(cases[case_key], defect.class_name, location_info["location"])
    st.sidebar.caption(cases[case_key]["description"])

    if "last_case_key" not in st.session_state or st.session_state.last_case_key != (image_id, obj_index, case_key):
        st.session_state.diagnosis_text = json.dumps(case_payload, ensure_ascii=False, indent=2)
        st.session_state.last_case_key = (image_id, obj_index, case_key)

    st.markdown(
        "#### 演示流程\n"
        "`WTBD 叶片图像 → 缺陷 bbox 与 patch → MLLM 结构化诊断 → AEGIS 属性级核查 → 错误属性定位与复核反馈`"
    )

    col1, col2, col3 = st.columns([1.1, 0.9, 1.0])

    with col1:
        st.subheader("1. WTBD 原始图像与 XML 框")
        st.image(
            draw_bbox(image, annotation.objects, selected_index=defect.index),
            caption=f"{annotation.image_id} | subset={split.get(annotation.image_id, 'demo')} | {annotation.width}×{annotation.height}",
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
        diagnosis_text = st.text_area("现场可修改类型和位置", key="diagnosis_text", height=270)
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

    st.dataframe(result_dataframe(results), use_container_width=True, hide_index=True)
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


if __name__ == "__main__":
    main()
