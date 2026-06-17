from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voc_reader import BBox


CLASS_MAP = {
    "craze": "网状裂纹",
    "corrosion": "腐蚀",
    "surface_injure": "表面损伤",
    "thunderstrike": "雷击损伤",
    "crack": "裂纹",
    "hide_craze": "隐性裂纹",
}

CLASS_ALIASES = {
    "网状裂纹": "craze",
    "龟裂": "craze",
    "腐蚀": "corrosion",
    "表面损伤": "surface_injure",
    "表面伤": "surface_injure",
    "损伤": "surface_injure",
    "雷击损伤": "thunderstrike",
    "雷击": "thunderstrike",
    "裂纹": "crack",
    "裂缝": "crack",
    "隐性裂纹": "hide_craze",
    "隐藏裂纹": "hide_craze",
}

GRID_ALIASES = {
    "左上": "图像左上区域",
    "中上": "图像中上区域",
    "右上": "图像右上区域",
    "左中": "图像左中区域",
    "中心": "图像中心区域",
    "中间": "图像中心区域",
    "右中": "图像右中区域",
    "左下": "图像左下区域",
    "中下": "图像中下区域",
    "右下": "图像右下区域",
}


@dataclass(frozen=True)
class AttributeResult:
    attribute: str
    diagnosis: str
    evidence: str
    result: str
    enabled: bool = True


def class_to_zh(class_name: str) -> str:
    return CLASS_MAP.get(class_name, class_name)


def normalize_class_name(value: str) -> str | None:
    value = str(value or "").strip()
    if not value:
        return None
    if value in CLASS_MAP:
        return value
    if value in CLASS_ALIASES:
        return CLASS_ALIASES[value]

    lowered = value.lower()
    for key in CLASS_MAP:
        if key in lowered:
            return key
    for zh, key in CLASS_ALIASES.items():
        if zh in value:
            return key
    return None


def normalize_location(value: str) -> str | None:
    value = str(value or "").strip().replace(" ", "")
    if not value:
        return None
    for short, full in GRID_ALIASES.items():
        if short in value:
            return full
    for full in GRID_ALIASES.values():
        if full in value:
            return full
    return None


def calculate_relative_location(bbox: BBox, image_width: int, image_height: int) -> dict[str, Any]:
    cx, cy = bbox.center
    rx = cx / image_width
    ry = cy / image_height
    col = "左" if rx < 1 / 3 else "中" if rx < 2 / 3 else "右"
    row = "上" if ry < 1 / 3 else "中" if ry < 2 / 3 else "下"
    label = "图像中心区域" if col == "中" and row == "中" else f"图像{col}{row}区域"
    return {"center_x": cx, "center_y": cy, "rx": rx, "ry": ry, "location": label}


def verify_type(diagnosis_type: str, xml_class: str) -> AttributeResult:
    pred = normalize_class_name(diagnosis_type)
    truth = normalize_class_name(xml_class)
    evidence = f"XML：{class_to_zh(xml_class)} ({xml_class})"
    if pred is None:
        result = "无法识别"
    elif pred == truth:
        result = "一致"
    else:
        result = "不一致"
    return AttributeResult("类型", diagnosis_type or "未填写", evidence, result)


def verify_location(diagnosis_location: str, bbox: BBox, image_width: int, image_height: int) -> AttributeResult:
    calc = calculate_relative_location(bbox, image_width, image_height)
    pred = normalize_location(diagnosis_location)
    evidence = (
        f"bbox中心=({calc['center_x']:.1f}, {calc['center_y']:.1f}), "
        f"rx={calc['rx']:.2f}, ry={calc['ry']:.2f}；{calc['location']}"
    )
    if pred is None:
        result = "无法识别"
    elif pred == calc["location"]:
        result = "一致"
    else:
        result = "不一致"
    return AttributeResult("位置", diagnosis_location or "未填写", evidence, result)


def disabled_result(attribute: str, diagnosis: str, reason: str) -> AttributeResult:
    return AttributeResult(attribute, diagnosis or "未填写", reason, "暂未启用", enabled=False)


def verify_diagnosis(diagnosis: dict[str, Any], xml_class: str, bbox: BBox, image_width: int, image_height: int) -> list[AttributeResult]:
    return [
        verify_type(str(diagnosis.get("类型", "")), xml_class),
        verify_location(str(diagnosis.get("位置", "")), bbox, image_width, image_height),
        disabled_result("严重度", str(diagnosis.get("严重度", "")), "当前 WTBD 数据集缺少专家严重度真值标签"),
        disabled_result("形态", str(diagnosis.get("形态", "")), "当前 WTBD 数据集缺少形态属性级标注"),
    ]


def generate_feedback(results: list[AttributeResult]) -> str:
    inconsistent = [r for r in results if r.enabled and r.result != "一致"]
    if not inconsistent:
        return "AEGIS 当前验证未发现类型或位置属性不一致；该诊断通过当前原型验证。"

    lines = ["AEGIS 发现以下需要复核的属性：", ""]
    for idx, item in enumerate(inconsistent, start=1):
        lines.append(f"{idx}. {item.attribute}属性{item.result}。")
        lines.append(f"   MLLM 描述：{item.diagnosis}")
        lines.append(f"   参考证据：{item.evidence}")
        lines.append("")
    lines.append("建议 MLLM 或人工专家优先重新检查上述属性描述。")
    return "\n".join(lines)


def overall_status(results: list[AttributeResult]) -> str:
    if any(r.enabled and r.result != "一致" for r in results):
        return "诊断报告存在属性不一致，需要复核。"
    return "诊断报告通过当前类型与位置验证。"
