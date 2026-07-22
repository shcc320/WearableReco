from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PAMAP2_URL = "https://archive.ics.uci.edu/static/public/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip"
MHEALTH_URL = "https://archive.ics.uci.edu/static/public/319/mhealth%2Bdataset.zip"

COMMON_ACTIVITIES = ["lying", "sitting", "standing", "walking", "running", "cycling"]

PAMAP2_LABELS = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
}
MHEALTH_LABELS = {
    1: "standing",
    2: "sitting",
    3: "lying",
    4: "walking",
    9: "cycling",
    10: "running",  # jogging is merged with running for cross-dataset alignment
    11: "running",
}

COMMON_CHANNELS = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
]

SENSOR_CONFIGS = {
    "Chest acceleration": ["chest_acc_"],
    "Ankle acceleration": ["ankle_acc_"],
    "Ankle acceleration and gyroscope": ["ankle_acc_", "ankle_gyro_"],
    "Chest and ankle acceleration": ["chest_acc_", "ankle_acc_"],
    "Full common configuration": ["chest_acc_", "ankle_acc_", "ankle_gyro_"],
}


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        if zipfile.is_zipfile(target):
            return
        # An interrupted download may leave a non-empty, unusable file behind.
        # Remove only that invalid archive so the next download can resume the
        # normal workflow instead of failing later during extraction.
        print(f"Removing invalid archive left by an interrupted download: {target}")
        target.unlink()
    print(f"Downloading {url}\n  -> {target}")
    urllib.request.urlretrieve(url, target)


def _extract(zip_path: Path, destination: Path) -> None:
    marker = destination / ".extracted"
    destination.mkdir(parents=True, exist_ok=True)

    # The PAMAP2 UCI download is a zip containing a second zip
    # (PAMAP2_Dataset.zip).  Extract the outer archive once, then unpack that
    # inner dataset archive whenever the expected Protocol files are absent.
    # Checking the files rather than trusting only a marker also repairs output
    # produced by earlier versions that stopped after the first extraction.
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(destination)

    protocol_files = list(destination.glob("**/Protocol/subject*.dat"))
    for nested_zip in destination.glob("**/PAMAP2_Dataset.zip"):
        if protocol_files:
            break
        with zipfile.ZipFile(nested_zip) as archive:
            archive.extractall(destination)
        protocol_files = list(destination.glob("**/Protocol/subject*.dat"))

    marker.write_text("ok", encoding="utf-8")


def download_datasets(raw_root: Path) -> None:
    pamap_zip = raw_root / "pamap2.zip"
    mhealth_zip = raw_root / "mhealth.zip"
    _download(PAMAP2_URL, pamap_zip)
    _download(MHEALTH_URL, mhealth_zip)
    _extract(pamap_zip, raw_root / "pamap2")
    _extract(mhealth_zip, raw_root / "mhealth")


def _subject_number(path: Path) -> int:
    match = re.search(r"subject(\d+)", path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot determine subject id from {path}")
    return int(match.group(1))


def load_pamap2_samples(raw_root: Path, target_hz: int = 50) -> pd.DataFrame:
    files = sorted((raw_root / "pamap2").glob("**/Protocol/subject*.dat"))
    if not files:
        raise FileNotFoundError("PAMAP2 protocol files were not found. Run scripts/download_datasets.py first.")

    selected = {
        1: "activity_id",
        21: "chest_acc_x", 22: "chest_acc_y", 23: "chest_acc_z",
        38: "ankle_acc_x", 39: "ankle_acc_y", 40: "ankle_acc_z",
        44: "ankle_gyro_x", 45: "ankle_gyro_y", 46: "ankle_gyro_z",
    }
    frames: list[pd.DataFrame] = []
    for path in files:
        print(f"Reading PAMAP2 {path.name}")
        raw = pd.read_csv(path, sep=r"\s+", header=None, usecols=sorted(selected))
        raw = raw.rename(columns=selected)
        raw["segment_id"] = raw["activity_id"].ne(raw["activity_id"].shift()).cumsum()
        raw = raw[raw["activity_id"].isin(PAMAP2_LABELS)].copy()
        raw["activity"] = raw["activity_id"].map(PAMAP2_LABELS)
        raw["subject"] = f"P{_subject_number(path):02d}"

        cleaned_segments: list[pd.DataFrame] = []
        for _, segment in raw.groupby("segment_id", sort=False):
            segment = segment.copy()
            segment[COMMON_CHANNELS] = segment[COMMON_CHANNELS].interpolate(
                method="linear", limit_direction="both"
            )
            segment = segment.dropna(subset=COMMON_CHANNELS)
            # PAMAP2 is 100 Hz; use deterministic decimation to the common 50 Hz rate.
            if target_hz == 50:
                segment = segment.iloc[::2].copy()
            cleaned_segments.append(segment)
        if not cleaned_segments:
            # PAMAP2 subject 109 is distributed in the Protocol directory but
            # contains no samples from the six activities shared with MHEALTH.
            # It cannot contribute to the prespecified LOSO protocol.
            print(
                f"Skipping PAMAP2 {path.name}: no complete samples for the "
                "requested common activity set."
            )
            continue
        subject_df = pd.concat(cleaned_segments, ignore_index=True)
        missing_activities = sorted(set(COMMON_ACTIVITIES) - set(subject_df["activity"].unique()))
        if missing_activities:
            # A LOSO fold must use the same six-class task as every other fold.
            # PAMAP2 subject 103 does not include running or cycling, so keeping
            # it would change the evaluated label set and distort macro metrics.
            print(
                f"Skipping PAMAP2 {path.name}: missing required activities "
                f"{missing_activities}."
            )
            continue
        frames.append(subject_df[["subject", "segment_id", "activity", *COMMON_CHANNELS]])
    if not frames:
        raise RuntimeError(
            "No PAMAP2 samples remained after selecting the common activities "
            "and required sensor channels."
        )
    return pd.concat(frames, ignore_index=True)


def load_mhealth_samples(raw_root: Path) -> pd.DataFrame:
    files = sorted((raw_root / "mhealth").glob("**/mHealth_subject*.log"))
    if not files:
        raise FileNotFoundError("MHEALTH log files were not found. Run scripts/download_datasets.py first.")

    # Zero-based columns from the official MHEALTH description.
    selected = {
        0: "chest_acc_x", 1: "chest_acc_y", 2: "chest_acc_z",
        5: "ankle_acc_x", 6: "ankle_acc_y", 7: "ankle_acc_z",
        8: "ankle_gyro_x", 9: "ankle_gyro_y", 10: "ankle_gyro_z",
        23: "activity_id",
    }
    frames: list[pd.DataFrame] = []
    for path in files:
        print(f"Reading MHEALTH {path.name}")
        raw = pd.read_csv(path, sep=r"\s+", header=None, usecols=sorted(selected))
        raw = raw.rename(columns=selected)
        raw["segment_id"] = raw["activity_id"].ne(raw["activity_id"].shift()).cumsum()
        raw = raw[raw["activity_id"].isin(MHEALTH_LABELS)].copy()
        raw["activity"] = raw["activity_id"].map(MHEALTH_LABELS)
        raw["subject"] = f"M{_subject_number(path):02d}"
        frames.append(raw[["subject", "segment_id", "activity", *COMMON_CHANNELS]])
    return pd.concat(frames, ignore_index=True)


def _signal_features(signal: np.ndarray, sampling_hz: int) -> dict[str, float]:
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.mean(signal)
    rms = float(np.sqrt(np.mean(signal ** 2)))
    energy = float(np.mean(signal ** 2))
    zcr = float(np.mean(np.diff(np.signbit(centered)).astype(float)))

    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sampling_hz)
    if len(spectrum) > 1:
        spectrum[0] = 0.0
    total = float(spectrum.sum())
    if total > 0:
        probability = spectrum / total
        spectral_entropy = float(-np.sum(probability * np.log(probability + 1e-12)) / np.log(len(probability) + 1e-12))
        dominant_frequency = float(frequencies[int(np.argmax(spectrum))])
    else:
        spectral_entropy = 0.0
        dominant_frequency = 0.0

    q75, q25 = np.percentile(signal, [75, 25])
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal, ddof=1)),
        "median": float(np.median(signal)),
        "iqr": float(q75 - q25),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "rms": rms,
        "mean_abs": float(np.mean(np.abs(signal))),
        "energy": energy,
        "zcr": zcr,
        "dominant_frequency": dominant_frequency,
        "spectral_entropy": spectral_entropy,
    }


def extract_window_features(
    samples: pd.DataFrame,
    sampling_hz: int = 50,
    window_samples: int = 128,
    step_samples: int = 64,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    triads = {
        "chest_acc": ["chest_acc_x", "chest_acc_y", "chest_acc_z"],
        "ankle_acc": ["ankle_acc_x", "ankle_acc_y", "ankle_acc_z"],
        "ankle_gyro": ["ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"],
    }

    for (subject, segment_id, activity), segment in samples.groupby(
        ["subject", "segment_id", "activity"], sort=False
    ):
        values = segment[COMMON_CHANNELS].to_numpy(dtype=float)
        if len(values) < window_samples:
            continue
        for start in range(0, len(values) - window_samples + 1, step_samples):
            stop = start + window_samples
            window = segment.iloc[start:stop]
            row: dict[str, float | str | int] = {
                "subject": str(subject),
                "segment_id": int(segment_id),
                "activity": str(activity),
                "window_start": int(start),
            }
            for channel in COMMON_CHANNELS:
                for statistic, value in _signal_features(window[channel].to_numpy(), sampling_hz).items():
                    row[f"{channel}__{statistic}"] = value
            for group_name, channels in triads.items():
                magnitude = np.linalg.norm(window[channels].to_numpy(dtype=float), axis=1)
                for statistic, value in _signal_features(magnitude, sampling_hz).items():
                    row[f"{group_name}_mag__{statistic}"] = value
            rows.append(row)
    if not rows:
        raise RuntimeError("No windows were produced. Check the data files and window settings.")
    return pd.DataFrame(rows)


def feature_columns_for_configuration(frame: pd.DataFrame, configuration: str) -> list[str]:
    prefixes = SENSOR_CONFIGS[configuration]
    return sorted(
        column for column in frame.columns
        if "__" in column and any(column.startswith(prefix) for prefix in prefixes)
    )


def _primary_cache_matches_protocol(frame: pd.DataFrame) -> bool:
    """Return whether every cached primary subject has all study classes."""
    if frame.empty or not {"subject", "activity"}.issubset(frame.columns):
        return False
    required = set(COMMON_ACTIVITIES)
    return all(
        required.issubset(set(subject_frame["activity"].unique()))
        for _, subject_frame in frame.groupby("subject")
    )


def _keep_complete_primary_feature_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only subjects represented in every class after windowing."""
    required = set(COMMON_ACTIVITIES)
    eligible_subjects: list[str] = []
    for subject, subject_frame in frame.groupby("subject", sort=False):
        missing_activities = sorted(required - set(subject_frame["activity"].unique()))
        if missing_activities:
            print(
                f"Skipping PAMAP2 {subject} after windowing: missing required "
                f"activities {missing_activities}."
            )
        else:
            eligible_subjects.append(str(subject))
    if not eligible_subjects:
        raise RuntimeError("No PAMAP2 subjects have windows for all six required activities.")
    return frame[frame["subject"].isin(eligible_subjects)].copy()


def prepare_feature_datasets(
    raw_root: Path,
    processed_root: Path,
    target_hz: int,
    window_samples: int,
    step_samples: int,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_root.mkdir(parents=True, exist_ok=True)
    pamap_cache = processed_root / "pamap2_features.pkl.gz"
    mhealth_cache = processed_root / "mhealth_features.pkl.gz"

    rebuild_pamap = force or not pamap_cache.exists()
    if not rebuild_pamap:
        pamap_features = pd.read_pickle(pamap_cache, compression="gzip")
        if not _primary_cache_matches_protocol(pamap_features):
            print("Rebuilding PAMAP2 feature cache: it does not match the six-class protocol.")
            rebuild_pamap = True

    if rebuild_pamap:
        pamap_samples = load_pamap2_samples(raw_root, target_hz=target_hz)
        pamap_features = extract_window_features(
            pamap_samples, target_hz, window_samples, step_samples
        )
        pamap_features = _keep_complete_primary_feature_subjects(pamap_features)
        pamap_features.to_pickle(pamap_cache, compression="gzip")

    if force or not mhealth_cache.exists():
        mhealth_samples = load_mhealth_samples(raw_root)
        mhealth_features = extract_window_features(
            mhealth_samples, target_hz, window_samples, step_samples
        )
        mhealth_features.to_pickle(mhealth_cache, compression="gzip")
    else:
        mhealth_features = pd.read_pickle(mhealth_cache, compression="gzip")

    summary = []
    for name, frame in [("PAMAP2", pamap_features), ("MHEALTH", mhealth_features)]:
        for activity, count in frame["activity"].value_counts().sort_index().items():
            summary.append({"Dataset": name, "Activity": activity, "Windows": int(count)})
    pd.DataFrame(summary).to_csv(processed_root / "window_counts.csv", index=False)
    return pamap_features, mhealth_features


def make_synthetic_feature_datasets(seed: int = 47) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    feature_names = []
    for prefix in ["chest_acc", "ankle_acc", "ankle_gyro"]:
        for axis in ["x", "y", "z", "mag"]:
            for statistic in ["mean", "std", "rms", "energy", "dominant_frequency", "spectral_entropy"]:
                feature_names.append(f"{prefix}_{axis}__{statistic}")

    def build(dataset_prefix: str, n_subjects: int, shift: float) -> pd.DataFrame:
        rows = []
        for subject_idx in range(1, n_subjects + 1):
            subject_effect = rng.normal(0, 0.15, len(feature_names))
            for activity_idx, activity in enumerate(COMMON_ACTIVITIES):
                activity_effect = (activity_idx - 2.5) * 0.35
                for window_idx in range(8):
                    values = rng.normal(activity_effect + shift, 0.65, len(feature_names)) + subject_effect
                    row = {
                        "subject": f"{dataset_prefix}{subject_idx:02d}",
                        "segment_id": activity_idx + 1,
                        "activity": activity,
                        "window_start": window_idx * 64,
                    }
                    row.update(dict(zip(feature_names, values)))
                    rows.append(row)
        return pd.DataFrame(rows)

    return build("P", 4, 0.0), build("M", 3, 0.12)
