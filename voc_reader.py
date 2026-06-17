from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


DATASET_DIR = (
    Path(__file__).resolve().parent
    / "sample_data"
)


@dataclass(frozen=True)
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax


@dataclass(frozen=True)
class DefectObject:
    index: int
    class_name: str
    bbox: BBox
    pose: str = "Unspecified"
    truncated: int = 0
    difficult: int = 0


@dataclass(frozen=True)
class Annotation:
    image_id: str
    image_path: Path
    xml_path: Path
    width: int
    height: int
    depth: int
    objects: list[DefectObject]


def dataset_exists(dataset_dir: Path = DATASET_DIR) -> bool:
    return (dataset_dir / "JPEGImages").is_dir() and (dataset_dir / "Annotations").is_dir()


def list_image_ids(dataset_dir: Path = DATASET_DIR) -> list[str]:
    image_dir = dataset_dir / "JPEGImages"
    return sorted((p.name for p in image_dir.glob("*.jpg")), key=lambda name: int(Path(name).stem))


def read_split(dataset_dir: Path = DATASET_DIR) -> dict[str, str]:
    split_file = dataset_dir / "train_val_test_split.txt"
    if not split_file.exists():
        return {}

    rows: dict[str, str] = {}
    for line in split_file.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        image_id, subset = line.split(",", 1)
        rows[image_id.strip()] = subset.strip()
    return rows


def parse_annotation(image_id: str, dataset_dir: Path = DATASET_DIR) -> Annotation:
    stem = Path(image_id).stem
    xml_path = dataset_dir / "Annotations" / f"{stem}.xml"
    image_path = dataset_dir / "JPEGImages" / f"{stem}.jpg"

    if not xml_path.exists():
        raise FileNotFoundError(f"Missing annotation XML: {xml_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"XML has no <size>: {xml_path}")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    depth = int(size.findtext("depth", "0"))

    objects: list[DefectObject] = []
    for idx, obj in enumerate(root.findall("object"), start=1):
        box = obj.find("bndbox")
        if box is None:
            continue
        bbox = BBox(
            xmin=int(float(box.findtext("xmin", "0"))),
            ymin=int(float(box.findtext("ymin", "0"))),
            xmax=int(float(box.findtext("xmax", "0"))),
            ymax=int(float(box.findtext("ymax", "0"))),
        )
        objects.append(
            DefectObject(
                index=idx,
                class_name=obj.findtext("name", "").strip(),
                bbox=bbox,
                pose=obj.findtext("pose", "Unspecified"),
                truncated=int(obj.findtext("truncated", "0")),
                difficult=int(obj.findtext("difficult", "0")),
            )
        )

    return Annotation(
        image_id=image_path.name,
        image_path=image_path,
        xml_path=xml_path,
        width=width,
        height=height,
        depth=depth,
        objects=objects,
    )


def load_image(annotation: Annotation) -> Image.Image:
    return Image.open(annotation.image_path).convert("RGB")


def crop_patch(image: Image.Image, bbox: BBox, padding: int = 8) -> Image.Image:
    w, h = image.size
    left = max(0, bbox.xmin - padding)
    upper = max(0, bbox.ymin - padding)
    right = min(w, bbox.xmax + padding)
    lower = min(h, bbox.ymax + padding)
    return image.crop((left, upper, right, lower))


def draw_bbox(
    image: Image.Image,
    objects: list[DefectObject],
    selected_index: int | None = None,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\Noto Sans SC Medium (TrueType).otf", 24)
    except OSError:
        font = ImageFont.load_default()

    for obj in objects:
        is_selected = selected_index is None or obj.index == selected_index
        color = "#e11d48" if is_selected else "#64748b"
        width = 5 if is_selected else 3
        draw.rectangle(obj.bbox.as_tuple(), outline=color, width=width)
        label = f"{obj.index}. {obj.class_name}"
        label_box = draw.textbbox((obj.bbox.xmin, obj.bbox.ymin), label, font=font)
        bg = (label_box[0] - 4, label_box[1] - 3, label_box[2] + 6, label_box[3] + 4)
        draw.rectangle(bg, fill=color)
        draw.text((obj.bbox.xmin + 2, obj.bbox.ymin), label, fill="white", font=font)
    return canvas


def class_counts(dataset_dir: Path = DATASET_DIR) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image_id in list_image_ids(dataset_dir):
        ann = parse_annotation(image_id, dataset_dir)
        for obj in ann.objects:
            counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
    return dict(sorted(counts.items()))
