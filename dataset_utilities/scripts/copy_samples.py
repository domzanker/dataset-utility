from pathlib import Path
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

# print([t for t in args.path.rglob("cam_front.png")])
for path in args.path.rglob("cam_front.png"):
    out_path = args.out / path.relative_to(args.path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(path, out_path)
