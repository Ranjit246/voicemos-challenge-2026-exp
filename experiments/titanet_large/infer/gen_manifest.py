"""
Generate NeMo manifest.json for TitaNet embedding extraction.
Collects all unique wav paths from train.csv and dev.csv.
"""

import csv
import json
import os

WAV_ROOT = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn"
TRAIN_CSV = f"{WAV_ROOT}/sets/train.csv"
DEV_CSV   = f"{WAV_ROOT}/sets/dev.csv"
OUT_MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")

paths = set()
for csv_path in [TRAIN_CSV, DEV_CSV]:
    for row in csv.DictReader(open(csv_path)):
        paths.add(row["wav_a_path"])
        paths.add(row["wav_b_path"])

with open(OUT_MANIFEST, "w") as f:
    for rel_path in sorted(paths):
        abs_path = os.path.join(WAV_ROOT, rel_path)
        entry = {
            "audio_filepath": abs_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
        }
        f.write(json.dumps(entry) + "\n")

print(f"Written {len(paths)} entries to {OUT_MANIFEST}")
