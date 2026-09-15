"""
Pre-generates a multi-label training dataset for the MAD project on disk:
actual mixed/clean .wav files + a metadata CSV with multi-hot labels.
No runtime mixing logic is needed by the training DataLoader afterwards.

Usage:
    .\venv\Scripts\python.exe script/generate_multilable_dataset.py

Design note:
  - Reserves a distinct, non-overlapping slice of source files for "clean" outputs (each copied as-is, 1:1).
  - Draws the remaining "2-class" / "3-class" mixture members by sampling (without replacement within a single mix,
    with replacement across the whole dataset) from the per-class pools.
  - Total output count matches the ~6,429-file training set.
"""

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# ── Paths (hardcoded) ───────────────────────────────────────────────────────
_cwd = Path.cwd()
ROOT = _cwd.parent
DATASET_ROOT = ROOT / 'data' / "processed_mad"
METADATA_CSV = DATASET_ROOT / "metadata.csv"
DATA_DIR = DATASET_ROOT / 'audio'
DATASET_OUTPUT_DIR = ROOT / 'data' / "multilabel_mad"
LOG_DIR = ROOT / 'outputs' / 'logs'
LOG_FILE = LOG_DIR / "training.log"

# ── Generation config ───────────────────────────────────────────────────────
SEED          = 42
CLEAN_RATIO   = 0.50
TWO_CLASS     = 0.35
THREE_CLASS   = 0.15
VAL_RATIO     = 0.15
ALPHA_LOW     = 0.3
ALPHA_HIGH    = 0.7
SAMPLE_RATE   = 16_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _validate_ratios():
    ratio_sum = CLEAN_RATIO + TWO_CLASS + THREE_CLASS
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        log.error("CLEAN_RATIO + TWO_CLASS + THREE_CLASS must sum to 1.0 (got %.4f)", ratio_sum)
        sys.exit(1)


def load_source_metadata(source_csv: str) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    required_cols = {"id", "file", "label", "label_name", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"source_csv is missing required columns: {missing}")
    return df


def build_class_pools(train_df: pd.DataFrame):
    """label_id -> list of dict(file, label, label_name) for that class."""
    pools = {}
    for label_id, group in train_df.groupby("label"):
        pools[int(label_id)] = group[["file", "label", "label_name"]].to_dict("records")
    return pools


def peak_normalize_int16(mixed_float: np.ndarray) -> np.ndarray:
    """Peak-normalize a float waveform to stay within int16 range, then clip/cast."""
    peak = np.max(np.abs(mixed_float))
    if peak > 32767:
        mixed_float = mixed_float * (32767.0 / peak)
    mixed_float = np.clip(mixed_float, -32768, 32767)
    return mixed_float.astype(np.int16)


def read_wav_int16(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="int16")
    return data


def make_clean(record, source_audio_dir: Path):
    audio = read_wav_int16(source_audio_dir / record["file"])
    return audio, [record["label"]], [record["label_name"]], [record["file"]], [1.0]


def make_two_class_mix(class_pools, rng: np.random.Generator, source_audio_dir: Path):
    class_ids = rng.choice(list(class_pools.keys()), size=2, replace=False)
    rec_a = class_pools[class_ids[0]][rng.integers(0, len(class_pools[class_ids[0]]))]
    rec_b = class_pools[class_ids[1]][rng.integers(0, len(class_pools[class_ids[1]]))]

    wav_a = read_wav_int16(source_audio_dir / rec_a["file"]).astype(np.float64)
    wav_b = read_wav_int16(source_audio_dir / rec_b["file"]).astype(np.float64)

    alpha = rng.uniform(ALPHA_LOW, ALPHA_HIGH)
    mixed = alpha * wav_a + (1 - alpha) * wav_b
    mixed = peak_normalize_int16(mixed)

    labels = [rec_a["label"], rec_b["label"]]
    label_names = [rec_a["label_name"], rec_b["label_name"]]
    source_files = [rec_a["file"], rec_b["file"]]
    weights = [round(float(alpha), 4), round(float(1 - alpha), 4)]
    return mixed, labels, label_names, source_files, weights


def make_three_class_mix(class_pools, rng: np.random.Generator, source_audio_dir: Path):
    class_ids = rng.choice(list(class_pools.keys()), size=3, replace=False)
    recs = [class_pools[cid][rng.integers(0, len(class_pools[cid]))] for cid in class_ids]

    wavs = [read_wav_int16(source_audio_dir / r["file"]).astype(np.float64) for r in recs]

    # Dirichlet sampling with a floor of 0.2 per source, renormalized.
    raw_weights = rng.dirichlet([1, 1, 1])
    floor = 0.2
    weights = raw_weights * (1 - 3 * floor) + floor
    weights = weights / weights.sum()

    mixed = sum(w * wav for w, wav in zip(weights, wavs))
    mixed = peak_normalize_int16(mixed)

    labels = [r["label"] for r in recs]
    label_names = [r["label_name"] for r in recs]
    source_files = [r["file"] for r in recs]
    weight_list = [round(float(w), 4) for w in weights]
    return mixed, labels, label_names, source_files, weight_list


def to_multihot(labels, num_classes):
    vec = [0] * num_classes
    for l in labels:
        vec[int(l)] = 1
    return vec


def stratified_train_val_split(df: pd.DataFrame, val_ratio: float, rng: np.random.Generator):
    """Split rows into train/val, stratified by mix_type."""
    split_col = pd.Series(index=df.index, dtype=object)
    for mix_type, group in df.groupby("mix_type"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        split_col.loc[val_idx] = "val"
        split_col.loc[train_idx] = "train"
    return split_col


def main():
    _validate_ratios()
    rng = np.random.default_rng(SEED)

    output_audio_dir = DATASET_OUTPUT_DIR / "audio"
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading source metadata from %s", METADATA_CSV)
    meta = load_source_metadata(str(METADATA_CSV))
    train_df = meta[meta["split"] == "training"].reset_index(drop=True)
    if len(train_df) == 0:
        log.error("No rows found with split == 'training'. Check METADATA_CSV / column values.")
        sys.exit(1)

    label_name_by_id = (
        train_df[["label", "label_name"]].drop_duplicates().set_index("label")["label_name"].to_dict()
    )
    num_classes = int(train_df["label"].max()) + 1
    log.info("Loaded %d training source files across %d classes.", len(train_df), num_classes)

    class_pools = build_class_pools(train_df)

    total = len(train_df)
    n_clean = int(round(total * CLEAN_RATIO))
    n_two = int(round(total * TWO_CLASS))
    n_three = total - n_clean - n_two  # remainder absorbs rounding, keeps total exact

    log.info("Target counts -> clean: %d, 2-class: %d, 3-class: %d (total: %d)", n_clean, n_two, n_three, total)

    # Reserve a distinct slice of source files for "clean" outputs (true 1:1).
    shuffled_records = train_df.sample(frac=1.0, random_state=SEED).to_dict("records")
    clean_records = shuffled_records[:n_clean]

    output_rows = []
    out_id = 1

    # --- Clean outputs ---
    for rec in clean_records:
        audio, labels, label_names, source_files, weights = make_clean(rec, DATA_DIR)
        out_name = f"mad_ml_{out_id:06d}.wav"
        sf.write(str(output_audio_dir / out_name), audio, SAMPLE_RATE, subtype="PCM_16")
        output_rows.append(
            {
                "id": out_id,
                "file": out_name,
                "labels": ",".join(str(l) for l in labels),
                "label_names": ",".join(label_names),
                "label_vector": ",".join(str(v) for v in to_multihot(labels, num_classes)),
                "mix_type": "clean",
                "source_files": "|".join(source_files),
                "mix_weights": "|".join(str(w) for w in weights),
            }
        )
        out_id += 1

    # --- 2-class mixtures ---
    for _ in range(n_two):
        audio, labels, label_names, source_files, weights = make_two_class_mix(
            class_pools, rng, DATA_DIR
        )
        out_name = f"mad_ml_{out_id:06d}.wav"
        sf.write(str(output_audio_dir / out_name), audio, SAMPLE_RATE, subtype="PCM_16")
        output_rows.append(
            {
                "id": out_id,
                "file": out_name,
                "labels": ",".join(str(l) for l in labels),
                "label_names": ",".join(label_names),
                "label_vector": ",".join(str(v) for v in to_multihot(labels, num_classes)),
                "mix_type": "2-class",
                "source_files": "|".join(source_files),
                "mix_weights": "|".join(str(w) for w in weights),
            }
        )
        out_id += 1

    # --- 3-class mixtures ---
    for _ in range(n_three):
        audio, labels, label_names, source_files, weights = make_three_class_mix(class_pools, rng, DATA_DIR)
        out_name = f"mad_ml_{out_id:06d}.wav"
        sf.write(str(output_audio_dir / out_name), audio, SAMPLE_RATE, subtype="PCM_16")
        output_rows.append(
            {
                "id": out_id,
                "file": out_name,
                "labels": ",".join(str(l) for l in labels),
                "label_names": ",".join(label_names),
                "label_vector": ",".join(str(v) for v in to_multihot(labels, num_classes)),
                "mix_type": "3-class",
                "source_files": "|".join(source_files),
                "mix_weights": "|".join(str(w) for w in weights),
            }
        )
        out_id += 1

    out_df = pd.DataFrame(output_rows)

    # --- Stratified train/val split ---
    out_df["split"] = stratified_train_val_split(out_df, VAL_RATIO, rng)

    metadata_path = DATASET_OUTPUT_DIR / "metadata.csv"
    out_df.to_csv(metadata_path, index=False)
    log.info("Wrote metadata for %d samples to %s", len(out_df), metadata_path)

    # --- Config snapshot ---
    config = {
        "seed": SEED,
        "source_csv": str(METADATA_CSV),
        "source_audio": str(DATA_DIR),
        "output_dir": str(DATASET_OUTPUT_DIR),
        "clean_ratio": CLEAN_RATIO,
        "two_class_ratio": TWO_CLASS,
        "three_class_ratio": THREE_CLASS,
        "val_ratio": VAL_RATIO,
        "alpha_range": [ALPHA_LOW, ALPHA_HIGH],
        "sample_rate": SAMPLE_RATE,
        "num_classes": num_classes,
        "label_name_by_id": {int(k): v for k, v in label_name_by_id.items()},
        "counts": {"clean": n_clean, "2-class": n_two, "3-class": n_three, "total": total},
        "note": (
            "clean outputs use a distinct 1:1 slice of source files; 2-class/3-class "
            "mixture members are sampled per-class (without replacement within a mix, "
            "with replacement across the dataset) since each mix needs multiple distinct-"
            "class sources per output."
        ),
    }
    config_path = DATASET_OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Wrote config snapshot to %s", config_path)

    # --- Summary report ---
    print("\n" + "=" * 60)
    print("MAD MULTI-LABEL DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total samples generated : {len(out_df)}")
    print("\nPer mix-type counts:")
    print(out_df["mix_type"].value_counts().to_string())

    print("\nPer-class frequency across all multi-hot labels:")
    label_matrix = np.array([[int(v) for v in row.split(",")] for row in out_df["label_vector"]])
    for cid in range(num_classes):
        cname = label_name_by_id.get(cid, f"class_{cid}")
        print(f"  {cid} ({cname}): {int(label_matrix[:, cid].sum())}")

    print("\nTrain/Val split counts:")
    print(out_df["split"].value_counts().to_string())

    print("\nAudio sanity check on 10 random outputs:")
    sample_files = out_df["file"].sample(n=min(10, len(out_df)), random_state=SEED)
    for fname in sample_files:
        fpath = output_audio_dir / fname
        info = sf.info(str(fpath))
        duration = info.frames / info.samplerate
        data, sr = sf.read(str(fpath), dtype="int16")
        print(f"  {fname}: sr={sr}, duration={duration:.2f}s, dtype={data.dtype}, frames={info.frames}")

    print("=" * 60)
    print(f"Done. Output written to: {DATASET_OUTPUT_DIR.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()