"""Buat 5-fold cross-validation stratified dari seluruh gambar export YOLO.

Semua gambar (train + valid + test dari export) digabung, lalu dibagi 5 fold
dengan iterative stratification berdasarkan kehadiran kelas dan jumlah sel per
kelas di tiap gambar. Untuk iterasi i:
    test  = fold i
    valid = fold (i + 1) % 5     (dipakai untuk early stopping / pilih checkpoint)
    train = tiga fold lainnya

Pemasangan:
    pip install iterative-stratification numpy PyYAML

Pemakaian:
    python make_folds.py PATH_EXPORT [--k 5] [--seed 42]
    python make_folds.py PATH_EXPORT --assignment configs/folds_assignment.json
        (di mesin lain: pakai pembagian yang sudah ada, hanya bangun ulang path-nya;
         tidak perlu iterative-stratification)

Keluaran di PATH_EXPORT/folds/:
    fold_0/ ... fold_4/   masing-masing berisi train.txt, val.txt, test.txt, data.yaml
    folds_assignment.json  nama gambar -> nomor fold (simpan ini di repo!)
    folds_summary.csv      jumlah gambar dan sel per kelas per fold
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument(
    "--assignment",
    help="Path ke folds_assignment.json. Jika diberikan, pembagian fold TIDAK dihitung ulang: "
    "dibaca dari file ini dan hanya daftar path (train/val/test.txt, data.yaml) yang dibuat "
    "ulang di mesin ini (Colab, vast.ai, Jetson).",
)
args = ap.parse_args()

root = os.path.abspath(args.root)
k = args.k
cfg = yaml.safe_load(open(os.path.join(root, "data.yaml")))
names = cfg["names"]
names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
nc = len(names)

images = []
for s in ["train", "valid", "test"]:
    for ext in ("jpg", "jpeg", "png"):
        images += glob.glob(os.path.join(root, s, "images", f"*.{ext}"))
images = sorted(images)
print(f"Gambar ditemukan: {len(images)}, kelas: {nc}")


def label_path(img):
    d, fn = os.path.split(img)
    return os.path.join(os.path.dirname(d), "labels", os.path.splitext(fn)[0] + ".txt")


counts = np.zeros((len(images), nc), dtype=int)
for i, img in enumerate(images):
    for ln in open(label_path(img)):
        p = ln.split()
        if p:
            counts[i, int(p[0])] += 1

# Fitur stratifikasi: kehadiran kelas + penanda "banyak" (di atas median jumlah sel
# kelas itu pada gambar yang memuatnya), supaya gambar padat tersebar merata.
presence = (counts > 0).astype(int)
many = np.zeros_like(presence)
for c in range(nc):
    nz = counts[counts[:, c] > 0, c]
    if len(nz):
        many[:, c] = (counts[:, c] > np.median(nz)).astype(int)
y = np.hstack([presence, many])

if args.assignment:
    mapping = json.load(open(args.assignment))
    missing = [os.path.basename(im) for im in images if os.path.basename(im) not in mapping]
    if missing:
        raise SystemExit(f"{len(missing)} gambar tidak ada di assignment, contoh: {missing[:3]}")
    fold_of = np.array([mapping[os.path.basename(im)] for im in images], dtype=int)
    print(f"Pembagian fold dibaca dari {args.assignment} (tidak dihitung ulang)")
else:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
    fold_of = np.zeros(len(images), dtype=int)
    for f, (_, idx) in enumerate(mskf.split(np.zeros(len(images)), y)):
        fold_of[idx] = f

out = os.path.join(root, "folds")
os.makedirs(out, exist_ok=True)

for i in range(k):
    test_f, val_f = i, (i + 1) % k
    parts = {
        "train": [im for im, f in zip(images, fold_of) if f not in (test_f, val_f)],
        "val": [im for im, f in zip(images, fold_of) if f == val_f],
        "test": [im for im, f in zip(images, fold_of) if f == test_f],
    }
    d = os.path.join(out, f"fold_{i}")
    os.makedirs(d, exist_ok=True)
    for name, files in parts.items():
        with open(os.path.join(d, f"{name}.txt"), "w") as fh:
            fh.write("\n".join(files) + "\n")
    data = {
        "train": os.path.join(d, "train.txt"),
        "val": os.path.join(d, "val.txt"),
        "test": os.path.join(d, "test.txt"),
        "nc": nc,
        "names": names,
    }
    with open(os.path.join(d, "data.yaml"), "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    print(f"fold_{i}: train {len(parts['train'])} / val {len(parts['val'])} / test {len(parts['test'])} gambar")

if not args.assignment:
    with open(os.path.join(out, "folds_assignment.json"), "w") as fh:
        json.dump({os.path.basename(im): int(f) for im, f in zip(images, fold_of)}, fh, indent=1)

# Ringkasan per fold: sel (gambar) per kelas
print("\nSel / gambar per kelas di tiap fold (fold = bagian data yang menjadi test pada iterasi itu)")
print(f"{'kelas':<16}" + "".join(f"{'fold ' + str(f):>12}" for f in range(k)))
rows = []
for c, name in enumerate(names):
    cells = [int(counts[fold_of == f, c].sum()) for f in range(k)]
    imgs = [int(presence[fold_of == f, c].sum()) for f in range(k)]
    print(f"{name:<16}" + "".join(f"{cells[f]:>7}/{imgs[f]:<4}" for f in range(k)))
    rows.append([name] + [x for f in range(k) for x in (cells[f], imgs[f])])
    if min(imgs) < 3:
        print(f"   PERINGATAN: kelas {name} hanya {min(imgs)} gambar di salah satu fold")

with open(os.path.join(out, "folds_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["kelas"] + [f"fold{f}_{t}" for f in range(k) for t in ("sel", "gambar")])
    w.writerows(rows)
print(f"\nSelesai. Keluaran di: {out}")
