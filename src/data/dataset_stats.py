"""Cek jumlah kelas, format label, dan class balance dari export YOLO Roboflow.

Pemakaian:
    python dataset_stats.py PATH_FOLDER_EXPORT

Folder export harus berisi data.yaml dan subfolder train/valid/test
masing-masing dengan folder labels/.
"""
import collections
import glob
import os
import sys

import yaml

root = sys.argv[1]
cfg = yaml.safe_load(open(os.path.join(root, "data.yaml")))
names = cfg["names"]
names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
print(f"Jumlah kelas (data.yaml): {len(names)}")

splits = ["train", "valid", "test"]
inst = {s: collections.Counter() for s in splits}  # jumlah sel per kelas
imgs = {s: collections.Counter() for s in splits}  # jumlah gambar yang memuat kelas
fmt = collections.Counter()
n_img = {}
n_empty = {}

for s in splits:
    files = glob.glob(os.path.join(root, s, "labels", "*.txt"))
    n_img[s] = len(files)
    n_empty[s] = 0
    for f in files:
        seen = set()
        lines = [ln.split() for ln in open(f) if ln.strip()]
        if not lines:
            n_empty[s] += 1
        for p in lines:
            c = int(p[0])
            inst[s][c] += 1
            seen.add(c)
            fmt["polygon" if len(p) > 5 else "box"] += 1
        for c in seen:
            imgs[s][c] += 1

print(f"Format label: {dict(fmt)}  (polygon = bisa untuk segmentasi, box = deteksi saja)")
for s in splits:
    print(f"{s}: {n_img[s]} gambar, {n_empty[s]} tanpa label")

print()
print(f"{'kelas':<16}" + "".join(f"{s + ' (sel/gbr)':>18}" for s in splits) + f"{'total sel':>12}")
for i, name in enumerate(names):
    row = f"{name:<16}"
    for s in splits:
        row += f"{inst[s][i]:>8} /{imgs[s][i]:>4}   "
    row += f"{sum(inst[s][i] for s in splits):>10}"
    print(row)

total = {s: sum(inst[s].values()) for s in splits}
print(f"{'TOTAL':<16}" + "".join(f"{total[s]:>18}" for s in splits) + f"{sum(total.values()):>12}")
