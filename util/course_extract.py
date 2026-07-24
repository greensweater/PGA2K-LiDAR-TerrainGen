# course_extract.py

import json
import base64
import gzip
from pathlib import Path
import sys

def decompress_gzip(data: bytes) -> bytes:
    return gzip.decompress(data)

def extract_course_file(input_path: str, output_dir: str):
    input_path = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read outer JSON (UTF-16LE gzip)
    with gzip.open(input_path, 'rb') as f:
        raw_json = f.read()
        #print(raw_json[:4])

    # Decode the UTF-16LE text    # Handle BOM if present
    if raw_json.startswith(b'\xff\xfe'):
        text_json = raw_json[2:].decode('utf-16le')
    else:
        text_json = raw_json.decode('utf-16le')
    #text_json = raw_json.decode('utf-16le')
    meta = json.loads(text_json)

    # Extract and decode CourseDescription
    encoded_cd = meta["binaryData"]["CourseDescription"]
    cd_compressed = base64.b64decode(encoded_cd)
    cd_bytes = decompress_gzip(cd_compressed)

    # Handle BOM if present
    if cd_bytes.startswith(b'\xff\xfe'):
        cd_text = cd_bytes[2:].decode('utf-16le')
    else:
        cd_text = cd_bytes.decode('utf-16le')

    course_desc = json.loads(cd_text)

    # Save base CourseDescription and chunk subnodes
    cd_nodes_dir = output_path / "CourseDescription_nodes"
    cd_nodes_dir.mkdir(exist_ok=True)
    chunkable = {}
    for k, v in course_desc.items():
        if isinstance(v, (list, dict)) and len(v) > 0:
            chunkable[k] = v
    for k in chunkable:
        with open(cd_nodes_dir / f"{k}.json", "w", encoding="utf-8") as f:
            json.dump(course_desc[k], f, ensure_ascii=False, indent=2)
        del course_desc[k]

    with open(output_path / "CourseDescription.json", "w", encoding="utf-8") as f:
        json.dump(course_desc, f, ensure_ascii=False, indent=2)

    # Extract CourseMetadata if present
    encoded_cm = meta["binaryData"].get("CourseMetadata")
    if encoded_cm:
        cm_compressed = base64.b64decode(encoded_cm)
        cm_bytes = decompress_gzip(cm_compressed)
        if cm_bytes.startswith(b'\xff\xfe'):
            cm_text = cm_bytes[2:].decode('utf-16le')
        else:
            cm_text = cm_bytes.decode('utf-16le')
        with open(output_path / "CourseMetadata.json", "w", encoding="utf-8") as f:
            f.write(cm_text)

    # Extract Thumbnail if present
    encoded_thumb = meta["binaryData"].get("Thumbnail")
    if encoded_thumb:
        thumb_data = decompress_gzip(base64.b64decode(encoded_thumb))
        thumb_meta = json.loads(thumb_data.decode("utf-16"))
        image_data = base64.b64decode(thumb_meta["image"])
        with open(output_path / "Thumbnail_meta.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in thumb_meta.items() if k != "image"}, f, indent=2)
        with open(output_path / "Thumbnail.jpg", "wb") as f:
            f.write(image_data)

    # Save meta.json (without binaryData)
    del meta["binaryData"]
    with open(output_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: course_extract.py <input.course> <output_folder>")
        sys.exit(1)
    extract_course_file(sys.argv[1], sys.argv[2])
