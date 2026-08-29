#!/usr/bin/env python3
from pathlib import Path
import json
import csv
import argparse
from PIL import Image

SIZE = 2000
MAP_MIN = -1000
MAP_MAX = 1000

def in_bounds(pos):
    """Check whether a position dict lies within the playable 2000x2000 map."""
    try:
        x = float(pos.get("x", 0))
        z = float(pos.get("z", 0))
        return MAP_MIN <= x <= MAP_MAX and MAP_MIN <= z <= MAP_MAX
    except Exception:
        return False

def normalize_scale(scale):
    """Normalize scale to ensure x, y, z match the Y value."""
    if not isinstance(scale, dict):
        return scale
    y = float(scale.get("y", 1.0))
    scale["x"] = y
    scale["y"] = y
    scale["z"] = y
    return scale

def clean_items(items, mask_pixels=None):
    kept = []
    removed_count = 0
    normalized_count = 0
    masked_count = 0

    for obj in items:
        pos = obj.get("position", {})

        if not in_bounds(pos):
            removed_count += 1
            continue

        try:
            x = float(pos.get("x", 0))
            z = float(pos.get("z", 0))

            if mask_pixels is not None:
                px, pz = world_to_pixel_round(x, z)

                if (px, pz) not in mask_pixels:
                    masked_count += 1
                    continue

        except Exception:
            removed_count += 1
            continue

        sc = obj.get("scale", {})
        obj["scale"] = normalize_scale(sc)
        normalized_count += 1

        kept.append(obj)

    return kept, removed_count, normalized_count, masked_count

def load_mask_exact(mask_path: Path):
    im = Image.open(mask_path).convert('RGB')
    w, h = im.size

    if (w, h) != (SIZE, SIZE):
        raise SystemExit(f"Mask must be exactly {SIZE}x{SIZE}")

    data = im.load()
    flagged = set()

    for y in range(h):
        for x in range(w):
            if data[x, y] != (0,0,255):
                flagged.add((x, y))

    return flagged

def world_to_pixel_round(x, z):
    px = round(x - MAP_MIN)
    pz = round(MAP_MAX - z)
    return int(px), int(pz)

def sort_and_count_by_items(path: Path, dry_run: bool, do_csv: bool, mask: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path.name} does not contain a list at top level")

    total_removed = 0
    total_masked = 0
    total_normalized = 0
    enriched = []

    mask_pixels = None
    if mask:
        mask_pixels = load_mask_exact(mask)
        
    for e in data:
        key_path = e.get("Key", {}).get("path", "UNKNOWN_PATH")
        val = e.get("Value", {})
        items = val.get("items", [])
        clusters = val.get("clusters", [])
        splines = val.get("splines", [])

        # Clean items before counting
        cleaned_items, removed_count, normalized_count, masked_count = clean_items(items, mask_pixels)
        val["items"] = cleaned_items
        total_removed += removed_count
        total_masked += masked_count
        total_normalized += normalized_count

        count_items = len(cleaned_items)
        count_clusters = len(clusters)
        count_splines = len(splines)
        total = count_items + count_clusters + count_splines
        enriched.append((key_path, count_items, count_clusters, count_splines, total, e))

    print(f"\n--- Cleanup Summary ---")
    print(f"Removed out-of-bounds items: {total_removed}")
    print(f"Removed masked items: {total_masked}")
    print(f"Normalized scale values: {total_normalized}")

    # Sort descending by items count (primary), then total
    enriched.sort(key=lambda t: (t[1], t[4]), reverse=True)

    print(f"\n--- Node counts in {path.name} ---")
    print(f"{'Items':>6} {'Clusters':>9} {'Splines':>8} {'Total':>7}  Key.path")
    print("-" * 80)
    for key_path, ci, cc, cs, total, _ in enriched:
        print(f"{ci:6} {cc:9} {cs:8} {total:7}  {key_path}")

    # Optional CSV output
    if do_csv:
        csv_path = path.with_suffix(".node_counts.csv")
        with csv_path.open("w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Key.path", "Items", "Clusters", "Splines", "Total"])
            for key_path, ci, cc, cs, total, _ in enriched:
                writer.writerow([key_path, ci, cc, cs, total])
        print(f"\nWrote summary CSV → {csv_path}")

    if not dry_run:
        bak = path.with_suffix(path.suffix + ".bak.json")
        path.rename(bak)
        sorted_data = [e for *_, e in enriched]
        path.write_text(json.dumps(sorted_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSorted & cleaned JSON saved → {path}")
        print(f"Backup saved → {bak}")
    else:
        print("\n(Dry run: JSON not modified)")

def main():
    ap = argparse.ArgumentParser(
        description="Clean, normalize, sort, and summarize placedObjects JSON (userLayers2/3)."
    )
    ap.add_argument("file", type=Path, help="Path to placedObjects JSON file")
    ap.add_argument("--dry", action="store_true", help="Dry run (don't modify JSON)")
    ap.add_argument("--csv", action="store_true", help="Write CSV summary")
    ap.add_argument('--mask', type=Path, help="Path to object mask")
    args = ap.parse_args()

    if not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")
    if args.mask and not args.mask.exists():
        raise SystemExit(f"File not found: {args.mask}")

    sort_and_count_by_items(args.file, dry_run=args.dry, do_csv=args.csv, mask=args.mask)

if __name__ == "__main__":
    main()
