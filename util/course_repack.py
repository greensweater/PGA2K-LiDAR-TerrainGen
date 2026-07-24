# course_repack.py

import json
import base64
import gzip
from pathlib import Path
import sys

def compress_gzip(data: bytes) -> bytes:
    out = gzip.compress(data)
    return out

def repack_course_file(input_dir: str, output_path: str):
    input_path = Path(input_dir)
    output_path = Path(output_path)

    # Load meta
    with open(input_path / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    binary_data = {}

    # Rebuild CourseDescription from base + subnodes
    course_desc_root_file = input_path / "CourseDescription.json"
    nodes_dir = input_path / "CourseDescription_nodes"
    if not course_desc_root_file.exists():
        raise FileNotFoundError("CourseDescription.json not found")

    with open(course_desc_root_file, "r", encoding="utf-8") as f:
        course_desc = json.load(f)

    if nodes_dir.exists():
        for node_file in sorted(nodes_dir.glob("*.json")):
            if ".bak" in node_file.name:
                continue  # skip backup files
            node_name = node_file.stem
            with open(node_file, "r", encoding="utf-8") as nf:
                course_desc[node_name] = json.load(nf)

    cd_text = json.dumps(course_desc, ensure_ascii=False)
    cd_bytes = cd_text.encode('utf-16le')
    cd_bom_bytes = b'\xff\xfe' + cd_bytes
    cd_compressed = compress_gzip(cd_bom_bytes)
    binary_data["CourseDescription"] = base64.b64encode(cd_compressed).decode("ascii")

    # CourseMetadata
    cmd_file = input_path / "CourseMetadata.json"
    if cmd_file.exists():
        with open(cmd_file, "r", encoding="utf-8") as f:
            cmd_text = f.read()
        cmd_bytes = cmd_text.encode('utf-16le')
        cmd_bom_bytes = b'\xff\xfe' + cmd_bytes
        cmd_compressed = compress_gzip(cmd_bom_bytes)
        binary_data["CourseMetadata"] = base64.b64encode(cmd_compressed).decode("ascii")

    # Thumbnail
    thumb_meta_file = input_path / "Thumbnail_meta.json"
    thumb_image_file = input_path / "Thumbnail.jpg"
    if thumb_meta_file.exists() and thumb_image_file.exists():
        with open(thumb_meta_file, "r", encoding="utf-8") as f:
            thumb_meta = json.load(f)
        with open(thumb_image_file, "rb") as f:
            image_data = f.read()
        thumb_meta["image"] = base64.b64encode(image_data).decode("ascii")
        thumb_bytes = json.dumps(thumb_meta, ensure_ascii=False).encode("utf-16le")
        thumb_compressed = compress_gzip(thumb_bytes)
        binary_data["Thumbnail"] = base64.b64encode(thumb_compressed).decode("ascii")

    meta["binaryData"] = binary_data

    # Dump full meta as UTF-16LE encoded JSON and compress
    full_json = json.dumps(meta, ensure_ascii=False).encode("utf-16le")
    with gzip.open(output_path, "wb") as f:
        f.write(full_json)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: course_repack.py <input_folder> <output.course>")
        sys.exit(1)
    repack_course_file(sys.argv[1], sys.argv[2])
