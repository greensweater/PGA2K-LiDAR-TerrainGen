import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True)
args = parser.parse_args()
folder = Path(args.folder)

for png_path in folder.glob("*.png"):
    try:
        im = np.array(Image.open(png_path).convert("L"))
        values = [int(im[255, x]) for x in range(256)]
        out_path = folder / f"{png_path.stem}.json"
        out_path.write_text(json.dumps(values))
        print(f"wrote {out_path}")
    except Exception:
        pass
