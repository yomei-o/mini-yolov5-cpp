"""Build a person/car dataset from coco128 for the scratch trainer.

Keeps only COCO classes person(0) and car(2), remaps them to {0:person, 1:car},
copies each qualifying image (JPG, read directly by the scratch trainer via
stb_image) plus its filtered label. The scratch trainer letterboxes internally,
so labels stay normalized to the original image.

    python tools/prep_coco_pb.py --src data/coco128 --out data/coco_pb
"""
import argparse
import glob
import os
import shutil

COCO_PERSON, COCO_CAR = 0, 2
REMAP = {COCO_PERSON: 0, COCO_CAR: 1}  # -> person, car


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/coco128")
    ap.add_argument("--out", default="data/coco_pb")
    args = ap.parse_args()

    img_dir = os.path.join(args.src, "images", "train2017")
    lbl_dir = os.path.join(args.src, "labels", "train2017")
    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    kept = 0
    counts = {0: 0, 1: 0}
    for jpg in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(jpg))[0]
        lbl = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl):
            continue
        lines = []
        for line in open(lbl):
            p = line.split()
            if not p:
                continue
            c = int(float(p[0]))
            if c in REMAP:
                lines.append(f"{REMAP[c]} {p[1]} {p[2]} {p[3]} {p[4]}")
                counts[REMAP[c]] += 1
        if not lines:
            continue  # skip images with no person/car
        # scratch reads JPG directly via stb_image now, so just copy the file
        shutil.copy(jpg, os.path.join(out_img, stem + ".jpg"))
        with open(os.path.join(out_lbl, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        kept += 1

    print(f"kept {kept} images -> {args.out}")
    print(f"  objects: person={counts[0]}, car={counts[1]}")


if __name__ == "__main__":
    main()
