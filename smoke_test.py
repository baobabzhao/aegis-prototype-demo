from __future__ import annotations

from verifier import calculate_relative_location, generate_feedback, verify_diagnosis
from voc_reader import dataset_exists, list_image_ids, parse_annotation


def main() -> None:
    print(f"dataset_exists={dataset_exists()}")
    image_ids = list_image_ids()
    print(f"sample_images={len(image_ids)}")
    image_id = "10.jpg" if "10.jpg" in image_ids else image_ids[0]
    annotation = parse_annotation(image_id)
    defect = annotation.objects[0]
    location = calculate_relative_location(defect.bbox, annotation.width, annotation.height)
    diagnosis = {
        "类型": "腐蚀",
        "位置": "图像左下区域",
        "严重度": "严重",
        "形态": "深色烧蚀并伴随局部材料剥离",
    }
    results = verify_diagnosis(
        diagnosis,
        defect.class_name,
        defect.bbox,
        annotation.width,
        annotation.height,
    )

    print(f"image={annotation.image_id}")
    print(f"objects={len(annotation.objects)}")
    print(f"selected_class={defect.class_name}")
    print(f"calculated_location={location['location']}")
    for item in results:
        print(f"{item.attribute}: {item.result} | {item.evidence}")
    print()
    print(generate_feedback(results))


if __name__ == "__main__":
    main()
