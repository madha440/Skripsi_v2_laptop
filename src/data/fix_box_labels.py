"""Cari (dan opsional konversi) label berformat box di dataset segmentasi YOLO.

Baris label segmentasi:  class x1 y1 x2 y2 x3 y3 ...   (minimal 3 titik = 7 angka)
Baris label box:         class xc yc w h               (5 angka)

Pemakaian:
    python fix_box_labels.py PATH_EXPORT            # hanya laporan (dry-run)
    python fix_box_labels.py PATH_EXPORT --convert  # ubah box jadi polygon persegi

Dengan --convert, file label yang diubah dicadangkan dulu ke PATH_EXPORT/_box_backup/.
Laporan CSV ditulis ke PATH_EXPORT/box_labels_report.csv.
"""
import argparse
import collections
import csv
import glob
import os
import shutil

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--convert", action="store_true")
args = ap.parse_args()

root = args.root
cfg = yaml.safe_load(open(os.path.join(root, "data.yaml")))
names = cfg["names"]
names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

splits = ["train", "valid", "test"]
rows = []  # satu baris per (split, file, kelas)
by_class = collections.Counter()
by_split = collections.Counter()
malformed = []
n_converted = 0


def clip(v):
    return min(1.0, max(0.0, v))


def box_to_polygon(cls, xc, yc, w, h):
    x1, y1, x2, y2 = clip(xc - w / 2), clip(yc - h / 2), clip(xc + w / 2), clip(yc + h / 2)
    pts = [x1, y1, x2, y1, x2, y2, x1, y2]
    return f"{cls} " + " ".join(f"{v:.6f}" for v in pts)


for s in splits:
    for f in sorted(glob.glob(os.path.join(root, s, "labels", "*.txt"))):
        out_lines = []
        file_box = collections.Counter()
        file_poly = collections.Counter()
        for ln in open(f).read().splitlines():
            p = ln.split()
            if not p:
                continue
            if len(p) == 5:
                c = int(p[0])
                file_box[c] += 1
                by_class[c] += 1
                by_split[s] += 1
                xc, yc, w, h = map(float, p[1:])
                out_lines.append(box_to_polygon(c, xc, yc, w, h))
            elif len(p) >= 7 and len(p) % 2 == 1:
                file_poly[int(p[0])] += 1
                out_lines.append(ln)
            else:
                malformed.append((s, os.path.basename(f), len(p)))
                out_lines.append(ln)
        for c in sorted(set(file_box) | set(file_poly)):
            if file_box[c]:
                rows.append((s, os.path.basename(f), names[c], file_box[c], file_poly[c]))
        if args.convert and file_box:
            backup_dir = os.path.join(root, "_box_backup", s)
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(f, os.path.join(backup_dir, os.path.basename(f)))
            with open(f, "w") as fh:
                fh.write("\n".join(out_lines) + "\n")
            n_converted += sum(file_box.values())

total = sum(by_class.values())
print(f"Baris berformat box: {total}")
print("Per split :", dict(by_split))
print("Per kelas :")
for c, n in by_class.most_common():
    print(f"  {names[c]:<16}{n:>6}")

per_file = collections.defaultdict(lambda: [0, 0])
for s, fn, _, nb, npoly in rows:
    per_file[(s, fn)][0] += nb
    per_file[(s, fn)][1] += npoly
print(f"\nGambar yang punya label box: {len(per_file)}")
print("Gambar dengan label box terbanyak:")
for (s, fn), (nb, _) in sorted(per_file.items(), key=lambda kv: -kv[1][0])[:15]:
    print(f"  {s:<6}{nb:>5}  {fn}")

if malformed:
    print(f"\nPERINGATAN: {len(malformed)} baris format tidak dikenali (jumlah angka bukan 5 atau ganjil >=7):")
    for item in malformed[:10]:
        print("  ", item)

report = os.path.join(root, "box_labels_report.csv")
with open(report, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["split", "file", "kelas", "baris_box", "baris_polygon_kelas_sama_di_file"])
    w.writerows(rows)
print(f"\nLaporan: {report}")

if args.convert:
    print(f"Dikonversi ke polygon persegi: {n_converted} baris (cadangan di _box_backup/)")
else:
    print("Dry-run: tidak ada file label yang diubah. Tambahkan --convert untuk mengonversi.")
